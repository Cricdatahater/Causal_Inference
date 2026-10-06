"""Repeated-seed experiments and business-assumption sensitivity for Stage 7."""
from __future__ import annotations

from itertools import product
import numpy as np
import pandas as pd
from causal_learners import MODEL_FEATURES, SLearner, TLearner
from doubly_robust import DRLearner, evaluate_observational_policy
from policies import top_score_policy, random_policy, oracle_policy, evaluate_policy
from simulation import simulate_customers

DEFAULT_SEEDS = tuple(range(42, 52))
RETENTION_VALUES = (40.0, 80.0, 120.0)
TREATMENT_COSTS = (5.0, 10.0, 20.0)
CAPACITIES = (0.10, 0.20, 0.30)
SCENARIO_COLUMNS = ["retention_value", "treatment_cost", "capacity"]


def business_scenarios(values=RETENTION_VALUES, costs=TREATMENT_COSTS, capacities=CAPACITIES):
    """Return a prespecified full factorial grid, rejecting duplicate scenarios."""
    rows = list(product(values, costs, capacities))
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("Scenario grid must be nonempty and contain no duplicates")
    result = pd.DataFrame(rows, columns=SCENARIO_COLUMNS, dtype=float)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Scenario settings must be finite")
    if (result[["retention_value", "treatment_cost"]] < 0).any().any():
        raise ValueError("Business values and costs must be nonnegative")
    if ((result.capacity < 0) | (result.capacity > 1)).any():
        raise ValueError("Capacity must be between zero and one")
    return result


def split_cohort(data, seed):
    """Split one cohort 70/15/15 without using treatment, outcomes, or truth."""
    if len(data) < 20:
        raise ValueError("At least 20 customers are required")
    order = np.random.default_rng(seed).permutation(len(data))
    training_end, calibration_end = int(0.70*len(data)), int(0.85*len(data))
    return tuple(data.iloc[index].reset_index(drop=True) for index in
                 (order[:training_end], order[training_end:calibration_end], order[calibration_end:]))


def calibrated_policy(calibration_cate, evaluation_cate, *, retention_value,
                      treatment_cost, capacity):
    """Calibrate an economic threshold on separate covariates, then enforce capacity.

    No outcomes or oracle columns are accepted. A no-treatment threshold is
    represented by None in metadata, avoiding Infinity in persisted JSON.
    """
    business_scenarios([retention_value], [treatment_cost], [capacity])
    cal, test = (np.asarray(values, dtype=float) for values in
                 (calibration_cate, evaluation_cate))
    if any(v.ndim != 1 or len(v) == 0 or not np.isfinite(v).all() for v in (cal, test)):
        raise ValueError("CATE predictions must be finite nonempty vectors")
    calibration_scores = retention_value*cal - treatment_cost
    selected = top_score_policy(calibration_scores, capacity=capacity, require_positive=True).astype(bool)
    scores = retention_value*test - treatment_cost
    threshold = float(calibration_scores[selected].min()) if selected.any() else None
    qualifies = (scores > 0) & (scores >= threshold) if threshold is not None else np.zeros(len(test), dtype=bool)
    decisions = top_score_policy(np.where(qualifies, scores, 0.0), capacity=capacity, require_positive=True)
    return decisions, {"threshold": threshold, "calibration_treated": int(selected.sum()),
                       "qualified_before_cap": int(qualifies.sum()),
                       "capacity_cap_active": bool(qualifies.sum() > int(np.floor(capacity*len(test))))}


def cate_metrics(name, prediction, truth):
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    return {"model": name, "true_ate": float(truth.mean()),
            "predicted_ate": float(prediction.mean()),
            "ate_error": float(abs(prediction.mean()-truth.mean())),
            "pehe": float(np.sqrt(np.mean((prediction-truth)**2))),
            "cate_mae": float(np.abs(prediction-truth).mean()),
            "cate_correlation": float(np.corrcoef(prediction, truth)[0, 1])}


def run_seed(seed, scenarios=None, *, population_size=20_000, model_seed=42):
    """Refit every learner on a new simulated cohort, then evaluate the entire grid.

    Algorithm seeds and model configurations stay fixed to isolate cohort/split
    variability. Business scenarios change decisions without retraining effects.
    """
    scenarios = business_scenarios() if scenarios is None else scenarios
    # Revalidate custom scenario tables rather than silently using bad settings.
    if scenarios.empty or scenarios.duplicated(SCENARIO_COLUMNS).any():
        raise ValueError("Scenarios must be nonempty and distinct")
    for row in scenarios.itertuples(index=False):
        business_scenarios([row.retention_value], [row.treatment_cost], [row.capacity])
    data = simulate_customers(number_of_customers=population_size, seed=seed)
    training, calibration, evaluation = split_cohort(data, seed)
    cohorts = [training, calibration, evaluation]
    assert all(set(cohorts[i].customer_id).isdisjoint(cohorts[j].customer_id)
               for i in range(3) for j in range(i+1, 3))
    observed_training = training[MODEL_FEATURES + ["treatment", "retained_90d"]]
    models = {"S-learner": SLearner(model_seed), "T-learner": TLearner(model_seed),
              "DR-learner": DRLearner(random_seed=model_seed, n_splits=3)}
    cal_predictions, test_predictions, effects = {}, {}, []
    for name, model in models.items():
        model.fit(observed_training)
        cal_predictions[name] = model.predict_cate(calibration[MODEL_FEATURES])
        test_predictions[name] = model.predict_cate(evaluation[MODEL_FEATURES])
        prediction = test_predictions[name]
        assert np.isfinite(prediction).all() and (np.abs(prediction) <= 1).all()
        effects.append({"seed": int(seed), **cate_metrics(name, prediction, evaluation.true_cate)})
    dr = models["DR-learner"]
    assert all(set(fit).isdisjoint(heldout) for fit, heldout in dr.fold_indices_)
    assert (dr.fold_assignments_ >= 0).all()
    nuisance = dr.predict_nuisance(evaluation[MODEL_FEATURES])
    diagnostic = {"seed": int(seed), "training_customers": len(training),
                  "calibration_customers": len(calibration), "evaluation_customers": len(evaluation),
                  "training_treatment_rate": float(training.treatment.mean()),
                  "evaluation_treatment_rate": float(evaluation.treatment.mean()),
                  "min_propensity": float(nuisance.propensity.min()),
                  "max_propensity": float(nuisance.propensity.max()),
                  "propensity_clipped_fraction": float(((nuisance.propensity < 0.05) | (nuisance.propensity > 0.95)).mean())}
    policy_rows = []
    for scenario in scenarios.to_dict("records"):
        value, cost, capacity = (float(scenario[key]) for key in SCENARIO_COLUMNS)
        decisions = {"Treat nobody": np.zeros(len(evaluation), dtype=int),
                     "Random": random_policy(len(evaluation), capacity=capacity, seed=seed)}
        threshold_info = {}
        for name in models:
            decisions[name], threshold_info[name] = calibrated_policy(
                cal_predictions[name], test_predictions[name],
                retention_value=value, treatment_cost=cost, capacity=capacity)
        decisions["Oracle"] = oracle_policy(evaluation, capacity=capacity,
                                            retention_value=value, treatment_cost=cost)
        truth_results = {name: evaluate_policy(evaluation, decision, name, capacity=capacity,
                                               retention_value=value, treatment_cost=cost)
                         for name, decision in decisions.items()}
        oracle_profit = truth_results["Oracle"]["incremental_profit"]
        for name, decision in decisions.items():
            assert decision.sum() <= int(np.floor(capacity*len(evaluation)))
            truth_result = truth_results[name]
            assert truth_result["incremental_profit"] <= oracle_profit + 1e-7
            row = {"seed": int(seed), **scenario, **truth_result,
                   "profit_per_customer": truth_result["incremental_profit"]/len(evaluation),
                   "regret_vs_oracle": oracle_profit-truth_result["incremental_profit"],
                   "oracle_profit": oracle_profit,
                   "oracle_profit_captured": (truth_result["incremental_profit"]/oracle_profit
                                               if oracle_profit > 1e-10 else None),
                   **threshold_info.get(name, {})}
            if name != "Oracle":
                observational = evaluate_observational_policy(
                    evaluation.retained_90d, evaluation.treatment, decision,
                    nuisance.mu0, nuisance.mu1, nuisance.propensity,
                    policy_name=name, retention_value=value, treatment_cost=cost)
                for key in ["dr_incremental_profit", "ci_lower_profit", "ci_upper_profit",
                            "standard_error_per_customer"]:
                    row[key] = observational[key]
                row["observational_estimation_error"] = row["dr_incremental_profit"] - row["incremental_profit"]
            policy_rows.append(row)
    return {"effects": effects, "policies": policy_rows, "diagnostics": [diagnostic]}


def summarize_replicates(frame, group_columns, metrics):
    """Summarize independent seeds, not pooled customers or dependent scenarios.

    SD describes between-seed variability. MCSE describes precision of the
    estimated seed-average. Quantiles are descriptive, not confidence intervals.
    """
    if frame.duplicated(group_columns + ["seed"]).any():
        raise ValueError("Each group must contain at most one row per seed")
    rows = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="raise").dropna()
            values = values.to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Summary values must be finite or missing")
            n = len(values)
            sd = float(values.std(ddof=1)) if n > 1 else None
            rows.append({**dict(zip(group_columns, keys)), "metric": metric,
                         "replicates": n, "mean": float(values.mean()) if n else None,
                         "std": sd, "mcse": sd/np.sqrt(n) if sd is not None else None,
                         "median": float(np.median(values)) if n else None,
                         "minimum": float(values.min()) if n else None,
                         "maximum": float(values.max()) if n else None,
                         "q025": float(np.quantile(values, 0.025)) if n else None,
                         "q975": float(np.quantile(values, 0.975)) if n else None})
    return pd.DataFrame(rows)


def paired_policy_differences(policies):
    """Use seed-paired oracle profit differences; report ties without choosing a winner."""
    index = ["seed", *SCENARIO_COLUMNS]
    if policies.duplicated(index + ["policy"]).any():
        raise ValueError("Duplicate seed/scenario/policy rows")
    table = policies.pivot(index=index, columns="policy", values="incremental_profit")
    rows = []
    for left, right in [("DR-learner", "S-learner"), ("DR-learner", "T-learner"),
                        ("S-learner", "T-learner")]:
        if left not in table or right not in table or table[[left, right]].isna().any().any():
            raise ValueError("Paired comparison requires all learner rows on identical seeds")
        for keys, delta in (table[left]-table[right]).items():
            rows.append({**dict(zip(index, keys)), "comparison": f"{left} minus {right}",
                         "profit_difference": float(delta),
                         "left_wins": float(delta > 1e-8), "tie": float(abs(delta) <= 1e-8)})
    return pd.DataFrame(rows)
