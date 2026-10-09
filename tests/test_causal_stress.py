"""Stress identification checks and experiment integrity."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import causal_stress
from causal_learners import MODEL_FEATURES
from doubly_robust import doubly_robust_scores


def test_perfect_observational_nuisances_cannot_fix_hidden_confounding():
    # Enumerate P(U,T,Y); U influences both assignment and potential outcomes.
    e = [0.2, 0.8]
    mu0, mu1 = [0.2, 0.8], [0.4, 1.0]
    observed_mu0 = sum((1-e[u])*mu0[u] for u in [0, 1])/sum(1-p for p in e)
    observed_mu1 = sum(e[u]*mu1[u] for u in [0, 1])/sum(e)
    y, t, weights = [], [], []
    for u in [0, 1]:
        for arm in [0, 1]:
            for outcome in [0, 1]:
                p = (mu1 if arm else mu0)[u]
                y.append(outcome); t.append(arm)
                weights.append(0.5*(e[u] if arm else 1-e[u])*(p if outcome else 1-p))
    _, _, score = doubly_robust_scores(y, t, [observed_mu0]*8,
                                      [observed_mu1]*8, [0.5]*8)
    causal_effect = np.mean(mu1)-np.mean(mu0)
    aipw_limit = np.array(weights) @ score
    assert aipw_limit == pytest.approx(observed_mu1-observed_mu0)
    assert causal_effect == pytest.approx(0.2)
    assert aipw_limit-causal_effect == pytest.approx(0.36)


def test_prespecified_scenarios_isolate_mechanisms():
    scenarios = causal_stress.STRESS_SCENARIOS
    assert len(scenarios) == 6 and len({row[0] for row in scenarios}) == 6
    assert scenarios[0] == ("Baseline", 1, 0)
    assert any(overlap > 1 and hidden == 0 for _, overlap, hidden in scenarios)
    assert any(overlap == 1 and hidden > 0 for _, overlap, hidden in scenarios)
    assert any(overlap > 1 and hidden > 0 for _, overlap, hidden in scenarios)


def test_paired_delta_requires_same_seed_and_model():
    frame = pd.DataFrame({"seed": [42, 42, 43, 43], "model": ["DR"]*4,
                          "scenario": ["Baseline", "Stress"]*2, "pehe": [.1, .3, .2, .25]})
    result = causal_stress.paired_baseline_deltas(frame, "pehe", ["model"])
    np.testing.assert_allclose(result.baseline_delta, [0, .2, 0, .05])
    with pytest.raises(ValueError):
        causal_stress.paired_baseline_deltas(frame.iloc[1:], "pehe", ["model"])
    with pytest.raises(ValueError):
        causal_stress.paired_baseline_deltas(pd.concat([frame, frame.iloc[:1]]), "pehe", ["model"])


@pytest.mark.parametrize("scenario", [causal_stress.STRESS_SCENARIOS[0], causal_stress.STRESS_SCENARIOS[4]])
def test_pipeline_excludes_latent_and_oracle_columns(monkeypatch, scenario):
    class HonestStub:
        def __init__(self, *args, **kwargs): pass
        def fit(self, data):
            assert set(data) == set(MODEL_FEATURES + ["treatment", "retained_90d"])
            self.fold_indices_ = []
            self.fold_assignments_ = np.zeros(len(data), dtype=int)
        def predict_cate(self, data):
            assert set(data) == set(MODEL_FEATURES)
            return .05 + data.login_days_30d.to_numpy()/100
        def predict_nuisance(self, data):
            assert set(data) == set(MODEL_FEATURES)
            return pd.DataFrame({"propensity": [.3]*len(data), "mu0": [.4]*len(data), "mu1": [.6]*len(data)})
    for name in ["SLearner", "TLearner", "DRLearner"]:
        monkeypatch.setattr(causal_stress, name, HonestStub)
    result = causal_stress.run_stress(42, scenario, population_size=1000)
    assert len(result["effects"]) == 3 and len(result["policies"]) == 6
    assert len(result["clipping"]) == 9 and len(result["overlap"]) == 80
    assert sum(row["count"] for row in result["overlap"]) == 2*150
    assert all(row["capacity_feasible"] for row in result["policies"])
    assert all(row["regret_vs_oracle"] >= -1e-7 for row in result["policies"])
    gap = result["diagnostics"][0]["analytic_identification_gap"]
    if scenario[2] == 0: assert gap == pytest.approx(0)
    else: assert gap > .1
    for row in result["policies"]:
        if row["policy"] == "Treat nobody":
            assert row["incremental_profit"] == row["aipw_profit"] == row["analytic_policy_identification_gap"] == 0
