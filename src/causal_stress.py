"""Paired causal-identification stress experiments for Stage 8."""
from __future__ import annotations
import numpy as np
import pandas as pd
from causal_learners import MODEL_FEATURES, SLearner, TLearner
from doubly_robust import DRLearner, doubly_robust_scores, evaluate_observational_policy
from robustness import split_cohort, calibrated_policy, cate_metrics
from policies import random_policy, oracle_policy, evaluate_policy
from stress_simulation import simulate_stress_customers, STRESS_ORACLE_COLUMNS

DEFAULT_SEEDS = (42, 43, 44, 45, 46)
STRESS_SCENARIOS = (
    ("Baseline", 1.0, 0.0), ("Moderate overlap", 2.5, 0.0),
    ("Severe overlap", 5.0, 0.0), ("Moderate hidden", 1.0, 0.8),
    ("Severe hidden", 1.0, 1.6), ("Combined severe", 5.0, 1.6),
)


def run_stress(seed, scenario, *, population_size=20_000, model_seed=42):
    name, overlap, hidden = scenario
    data = simulate_stress_customers(population_size, seed,
                                     overlap_strength=overlap, hidden_strength=hidden)
    training, calibration, evaluation = split_cohort(data, seed)
    assert set(training.customer_id).isdisjoint(evaluation.customer_id)
    assert set(calibration.customer_id).isdisjoint(evaluation.customer_id)
    assert set(training.customer_id).isdisjoint(calibration.customer_id)
    assert not set(MODEL_FEATURES) & set(STRESS_ORACLE_COLUMNS)
    models = {"S-learner": SLearner(model_seed), "T-learner": TLearner(model_seed),
              "DR-learner": DRLearner(random_seed=model_seed, n_splits=3)}
    effects, predictions, decisions, threshold_info = [], {}, {}, {}
    metadata = {"seed": int(seed), "scenario": name,
                "overlap_strength": overlap, "hidden_strength": hidden}
    for model_name, model in models.items():
        model.fit(training[MODEL_FEATURES + ["treatment", "retained_90d"]])
        prediction = model.predict_cate(evaluation[MODEL_FEATURES])
        predictions[model_name] = prediction
        assert np.isfinite(prediction).all() and (np.abs(prediction) <= 1).all()
        effects.append({**metadata, **cate_metrics(model_name, prediction, evaluation.true_cate)})
        decisions[model_name], threshold_info[model_name] = calibrated_policy(
            model.predict_cate(calibration[MODEL_FEATURES]), prediction,
            retention_value=80, treatment_cost=10, capacity=0.2)
    dr = models["DR-learner"]
    assert all(set(fit).isdisjoint(heldout) for fit, heldout in dr.fold_indices_)
    assert (dr.fold_assignments_ >= 0).all()
    nuisance = dr.predict_nuisance(evaluation[MODEL_FEATURES])
    y, t = evaluation.retained_90d, evaluation.treatment
    _, _, estimated_score = doubly_robust_scores(y, t, nuisance.mu0, nuisance.mu1, nuisance.propensity)
    _, _, reference_score = doubly_robust_scores(
        y, t, evaluation.observational_mu0, evaluation.observational_mu1,
        evaluation.true_propensity, propensity_clip=0.001)
    association = (evaluation.observational_mu1-evaluation.observational_mu0).to_numpy()
    ehat = nuisance.propensity.to_numpy()
    e = evaluation.true_propensity.to_numpy()
    weights = np.where(t == 1, 1/np.clip(ehat, .05, .95), 1/(1-np.clip(ehat, .05, .95)))
    trim = (ehat >= .05) & (ehat <= .95)
    diagnostic = {**metadata, "evaluation_customers": len(evaluation),
                  "treatment_rate": float(t.mean()),
                  "true_ate": float(evaluation.true_cate.mean()),
                  "analytic_association_ate": float(association.mean()),
                  "analytic_identification_gap": float(association.mean()-evaluation.true_cate.mean()),
                  "estimated_aipw_ate": float(estimated_score.mean()),
                  "reference_aipw_ate": float(reference_score.mean()),
                  "true_weak_overlap_fraction": float(((e < .05) | (e > .95)).mean()),
                  "estimated_weak_overlap_fraction": float((~trim).mean()),
                  "propensity_rmse": float(np.sqrt(np.mean((ehat-e)**2))),
                  "max_clipped_inverse_weight": float(weights.max()),
                  "clipped_weight_ess": float(weights.sum()**2/np.square(weights).sum()),
                  "trimmed_customers": int(trim.sum()),
                  "trimmed_true_ate": float(evaluation.true_cate[trim].mean()) if trim.any() else None,
                  "trimmed_estimated_aipw_ate": float(estimated_score[trim].mean()) if trim.any() else None}
    decisions = {"Treat nobody": np.zeros(len(evaluation), dtype=int),
                 "Random": random_policy(len(evaluation), .2, seed), **decisions,
                 "Oracle": oracle_policy(evaluation, .2)}
    true_results = {policy: evaluate_policy(evaluation, decision, policy, capacity=.2)
                    for policy, decision in decisions.items()}
    oracle_profit = true_results["Oracle"]["incremental_profit"]
    policy_rows, clipping_rows = [], []
    for policy, decision in decisions.items():
        truth = true_results[policy]
        assert truth["capacity_feasible"] and truth["incremental_profit"] <= oracle_profit+1e-7
        row = {**metadata, **truth, "oracle_profit": oracle_profit,
               "regret_vs_oracle": oracle_profit-truth["incremental_profit"],
               "oracle_profit_captured": truth["incremental_profit"]/oracle_profit if oracle_profit > 1e-10 else None,
               **threshold_info.get(policy, {})}
        if policy != "Oracle":
            estimated = evaluate_observational_policy(y, t, decision, nuisance.mu0, nuisance.mu1,
                                                       nuisance.propensity, policy_name=policy)
            reference = evaluate_observational_policy(y, t, decision, evaluation.observational_mu0,
                                                       evaluation.observational_mu1, evaluation.true_propensity,
                                                       policy_name=policy, propensity_clip=.001)
            row.update({"aipw_profit": estimated["dr_incremental_profit"],
                        "aipw_lower": estimated["ci_lower_profit"], "aipw_upper": estimated["ci_upper_profit"],
                        "estimated_profit_error": estimated["dr_incremental_profit"]-truth["incremental_profit"],
                        "reference_aipw_profit": reference["dr_incremental_profit"],
                        "analytic_association_profit": float(np.sum(decision*(80*association-10))),
                        "analytic_policy_identification_gap": float(np.sum(decision*80*(association-evaluation.true_cate)))})
        policy_rows.append(row)
        if policy in models:
            for clip in [.01, .05, .10]:
                result = evaluate_observational_policy(y, t, decision, nuisance.mu0, nuisance.mu1,
                                                        nuisance.propensity, policy_name=policy, propensity_clip=clip)
                clipping_rows.append({**metadata, "policy": policy, "propensity_clip": clip,
                                      "aipw_profit": result["dr_incremental_profit"],
                                      "true_profit": truth["incremental_profit"],
                                      "estimated_profit_error": result["dr_incremental_profit"]-truth["incremental_profit"],
                                      "maximum_inverse_weight": result["maximum_observed_inverse_weight"],
                                      "weight_ess": result["observed_weight_ess"]})
    # Do not publish latent U or customer-level records; diagnostic histogram bins only.
    overlap_rows = []
    bins = np.linspace(0, 1, 21)
    for kind, values in [("True marginal", e), ("Estimated", ehat)]:
        for arm in (0, 1):
            counts, _ = np.histogram(values[t.to_numpy() == arm], bins=bins)
            overlap_rows.extend({**metadata, "kind": kind, "treatment": arm,
                                 "bin_left": float(bins[i]), "bin_right": float(bins[i+1]),
                                 "count": int(count)} for i, count in enumerate(counts))
    return {"effects": effects, "policies": policy_rows, "diagnostics": [diagnostic],
            "clipping": clipping_rows, "overlap": overlap_rows}


def paired_baseline_deltas(frame, metric, group_columns):
    """Pair each stress scenario with the same seed/model/policy baseline.

    These are empirical changes, not causal effects of stress on a fixed model:
    each scenario refits models and can change the underlying causal target.
    """
    keys = ["seed", *group_columns]
    if frame.duplicated([*keys, "scenario"]).any():
        raise ValueError("Duplicate paired experiment rows")
    base = frame[frame.scenario == "Baseline"][[*keys, metric]].rename(columns={metric: "baseline_value"})
    result = frame.merge(base, on=keys, how="left", validate="many_to_one")
    if result.baseline_value.isna().any():
        raise ValueError("Each stress result needs a matching baseline")
    result["baseline_delta"] = result[metric]-result.baseline_value
    return result
