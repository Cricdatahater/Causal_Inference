"""Scientific and boundary checks for repeated-seed robustness analysis."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import robustness
from causal_learners import MODEL_FEATURES
from simulation import simulate_customers


def test_prespecified_full_factorial_grid():
    grid = robustness.business_scenarios()
    assert len(grid) == 27
    assert not grid.duplicated().any()
    assert ((grid.retention_value == 80) & (grid.treatment_cost == 10) & (grid.capacity == 0.2)).sum() == 1


@pytest.mark.parametrize("kwargs", [dict(values=[]), dict(values=[80, 80]),
                                    dict(costs=[-1]), dict(values=[np.nan]),
                                    dict(capacities=[1.1])])
def test_invalid_scenario_grids(kwargs):
    with pytest.raises(ValueError): robustness.business_scenarios(**kwargs)


def test_split_is_disjoint_deterministic_and_does_not_inspect_labels():
    data = simulate_customers(200, seed=42)
    first = robustness.split_cohort(data, 42)
    altered = data.copy()
    altered["retained_90d"] = -999
    altered["treatment"] = -999
    altered["true_cate"] = -999
    second = robustness.split_cohort(altered, 42)
    assert [len(frame) for frame in first] == [140, 30, 30]
    assert set.union(*(set(frame.customer_id) for frame in first)) == set(data.customer_id)
    for a, b in zip(first, second): np.testing.assert_array_equal(a.customer_id, b.customer_id)
    assert all(set(first[i].customer_id).isdisjoint(first[j].customer_id)
               for i in range(3) for j in range(i+1, 3))


@pytest.mark.parametrize("capacity", [0, 0.1, 0.2, 1])
def test_policy_capacity_and_profitability(capacity):
    calibration = np.linspace(-0.1, 0.4, 100)
    evaluation = np.linspace(-0.2, 0.5, 73)
    decisions, info = robustness.calibrated_policy(calibration, evaluation,
        retention_value=80, treatment_cost=10, capacity=capacity)
    assert decisions.sum() <= int(np.floor(capacity*len(evaluation)))
    assert ((80*evaluation[decisions.astype(bool)]-10) > 0).all()
    if capacity == 0:
        assert not decisions.any() and info["threshold"] is None


def test_no_profitable_candidates_and_strict_capacity_cap():
    decisions, info = robustness.calibrated_policy([0.1]*10, [0.5]*10,
        retention_value=80, treatment_cost=10, capacity=0.2)
    assert not decisions.any() and info["threshold"] is None
    decisions, info = robustness.calibrated_policy([0.3]*10, [0.5]*10,
        retention_value=80, treatment_cost=10, capacity=0.2)
    np.testing.assert_array_equal(decisions, [1, 1]+[0]*8)
    assert info["qualified_before_cap"] == 10 and info["capacity_cap_active"]


def test_high_cost_produces_no_treatment():
    decisions, _ = robustness.calibrated_policy([0.4]*20, [0.4]*20,
        retention_value=40, treatment_cost=20, capacity=0.3)
    assert not decisions.any()


def test_summary_uses_seed_replicates_and_sample_sd():
    frame = pd.DataFrame({"model": ["DR"]*3, "seed": [1, 2, 3], "profit": [10, 20, 30]})
    row = robustness.summarize_replicates(frame, ["model"], ["profit"]).iloc[0]
    assert row.replicates == 3 and row["mean"] == 20
    assert row["std"] == 10 and row.mcse == pytest.approx(10/np.sqrt(3))
    assert row.minimum == 10 and row.maximum == 30
    with pytest.raises(ValueError):
        robustness.summarize_replicates(pd.concat([frame, frame.iloc[:1]]), ["model"], ["profit"])


def test_zero_oracle_ratios_are_missing_not_infinite():
    frame = pd.DataFrame({"model": ["DR"]*2, "seed": [1, 2], "ratio": [None, 0.8]})
    row = robustness.summarize_replicates(frame, ["model"], ["ratio"]).iloc[0]
    assert row.replicates == 1 and row["mean"] == 0.8 and pd.isna(row["std"])


def test_seed_pairing_and_ties():
    rows = []
    for seed, amounts in [(1, [12, 10, 9]), (2, [20, 20, 19])]:
        for policy, amount in zip(["DR-learner", "S-learner", "T-learner"], amounts):
            rows.append(dict(seed=seed, retention_value=80, treatment_cost=10, capacity=0.2,
                             policy=policy, incremental_profit=amount))
    frame = pd.DataFrame(rows)
    result = robustness.paired_policy_differences(frame)
    pair = result[result.comparison == "DR-learner minus S-learner"]
    np.testing.assert_array_equal(pair.profit_difference, [2, 0])
    np.testing.assert_array_equal(pair.left_wins, [1, 0])
    np.testing.assert_array_equal(pair.tie, [0, 1])
    with pytest.raises(ValueError): robustness.paired_policy_differences(frame.iloc[:-1])


def test_experiment_uses_observed_training_and_covariate_only_prediction(monkeypatch):
    class HonestStub:
        def __init__(self, *args, **kwargs): pass
        def fit(self, data):
            assert set(data) == set(MODEL_FEATURES + ["treatment", "retained_90d"])
            self.fold_indices_ = []
            self.fold_assignments_ = np.zeros(len(data), dtype=int)
        def predict_cate(self, data):
            assert set(data) == set(MODEL_FEATURES)
            return 0.05 + data.login_days_30d.to_numpy()/100
        def predict_nuisance(self, data):
            assert set(data) == set(MODEL_FEATURES)
            return pd.DataFrame({"propensity": [0.3]*len(data), "mu0": [0.4]*len(data), "mu1": [0.6]*len(data)})
    for name in ["SLearner", "TLearner", "DRLearner"]:
        monkeypatch.setattr(robustness, name, HonestStub)
    result = robustness.run_seed(42, robustness.business_scenarios([40], [20], [0.2]), population_size=200)
    assert len(result["effects"]) == 3 and len(result["policies"]) == 6
    for row in result["policies"]:
        assert row["capacity_feasible"]
        assert row["incremental_profit"] <= row["oracle_profit"] + 1e-8
        if row["oracle_profit"] == 0: assert row["oracle_profit_captured"] is None
