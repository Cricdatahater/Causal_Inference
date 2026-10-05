"""Tests of DR algebra, honest folds, oracle exclusion, and policy evaluation."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from doubly_robust import DRLearner, doubly_robust_scores, evaluate_observational_policy
from simulation import simulate_customers


@pytest.mark.parametrize("correct", ["propensity", "outcomes"])
def test_double_robustness_population_identity(correct):
    # Enumerate the joint distribution of T and Y for a single covariate stratum.
    e, mu0, mu1 = 0.3, 0.4, 0.7
    t = np.array([0, 0, 1, 1])
    y = np.array([0, 1, 0, 1])
    weights = np.array([(1-e)*(1-mu0), (1-e)*mu0, e*(1-mu1), e*mu1])
    m0, m1 = (mu0, mu1) if correct == "outcomes" else (0.8, 0.2)
    estimated_e = e if correct == "propensity" else 0.6
    score0, score1, effect = doubly_robust_scores(
        y, t, np.full(4, m0), np.full(4, m1), np.full(4, estimated_e))
    assert weights @ score0 == pytest.approx(mu0)
    assert weights @ score1 == pytest.approx(mu1)
    assert weights @ effect == pytest.approx(mu1 - mu0)


def test_clipping_stabilizes_weights_without_clipping_scores():
    s0, s1, effect = doubly_robust_scores(
        [1, 0], [1, 0], [0.5, 0.5], [0.5, 0.5], [0, 1], propensity_clip=0.1)
    np.testing.assert_allclose(s0, [0.5, -4.5])
    np.testing.assert_allclose(s1, [5.5, 0.5])
    np.testing.assert_allclose(effect, [5, 5])


@pytest.fixture(scope="module")
def training():
    return simulate_customers(number_of_customers=600, seed=19)


@pytest.fixture(scope="module")
def fitted(training):
    return DRLearner(n_estimators=15, min_samples_leaf=10).fit(training)


def test_cross_fitting_covers_each_row_once_without_leakage(training, fitted):
    visited = np.zeros(len(training), dtype=int)
    for fold, (fit_indices, heldout_indices) in enumerate(fitted.fold_indices_):
        assert set(fit_indices).isdisjoint(heldout_indices)
        assert set(fit_indices) | set(heldout_indices) == set(range(len(training)))
        visited[heldout_indices] += 1
        np.testing.assert_array_equal(fitted.fold_assignments_[heldout_indices], fold)
    np.testing.assert_array_equal(visited, 1)
    assert np.isfinite(fitted.oof_nuisance_.to_numpy()).all()
    assert np.isfinite(fitted.pseudo_outcome_).all()


def test_training_ignores_oracle_and_post_treatment_columns(training, fitted):
    corrupted = training.copy()
    for name in ["customer_id", "true_cate", "true_mu0", "true_mu1", "true_propensity"]:
        corrupted[name] = -999
    corrupted["offer_redeemed"] = corrupted["retained_90d"]
    second = DRLearner(n_estimators=15, min_samples_leaf=10).fit(corrupted)
    new_data = simulate_customers(number_of_customers=100, seed=27)
    np.testing.assert_allclose(fitted.predict_cate(new_data), second.predict_cate(new_data))
    np.testing.assert_allclose(fitted.pseudo_outcome_, second.pseudo_outcome_)


def test_predictions_use_only_covariates_and_valid_probabilities(fitted):
    data = simulate_customers(number_of_customers=100, seed=27)
    from causal_learners import MODEL_FEATURES
    cate = fitted.predict_cate(data[MODEL_FEATURES])
    nuisance = fitted.predict_nuisance(data[MODEL_FEATURES])
    assert cate.shape == (100,)
    assert np.isfinite(cate).all() and (np.abs(cate) <= 1).all()
    assert nuisance.shape == (100, 3)
    assert ((nuisance >= 0) & (nuisance <= 1)).all().all()


def test_one_class_outcomes_are_supported(training):
    data = training.copy()
    data["retained_90d"] = 0
    learner = DRLearner(n_estimators=5).fit(data)
    np.testing.assert_allclose(learner.predict_cate(data.iloc[:10]), 0)
    np.testing.assert_allclose(learner.predict_nuisance(data.iloc[:10])[["mu0", "mu1"]], 0)


@pytest.mark.parametrize("change", ["missing", "nonbinary", "one_arm", "too_small"])
def test_invalid_training_fails(training, change):
    data = training.copy()
    if change == "missing":
        data = data.drop(columns="treatment")
    elif change == "nonbinary":
        data.loc[0, "retained_90d"] = 2
    elif change == "one_arm":
        data["treatment"] = 0
    else:
        data = data.iloc[:2]
    with pytest.raises(ValueError):
        DRLearner(n_estimators=5).fit(data)


def test_predict_before_fit_and_after_failed_refit(training):
    learner = DRLearner(n_estimators=5)
    with pytest.raises(RuntimeError):
        learner.predict_cate(training)
    learner.fit(training)
    with pytest.raises(ValueError):
        learner.fit(training.drop(columns="treatment"))
    with pytest.raises(RuntimeError):
        learner.predict_nuisance(training)


def test_paired_policy_contributions_and_zero_baseline():
    args = dict(outcome=[1, 0, 1, 0], treatment=[1, 0, 0, 1],
                mu0=[0.4]*4, mu1=[0.6]*4, propensity=[0.5]*4)
    baseline = evaluate_observational_policy(decisions=[0]*4, **args)
    assert baseline["dr_incremental_profit"] == 0
    assert baseline["ci_lower_profit"] == baseline["ci_upper_profit"] == 0
    policy = evaluate_observational_policy(decisions=[1, 0, 1, 0], **args)
    _, _, tau = doubly_robust_scores(**args)
    contributions = np.array([1, 0, 1, 0]) * (80 * tau - 10)
    assert policy["dr_incremental_profit"] == pytest.approx(contributions.sum())
    assert policy["standard_error_per_customer"] == pytest.approx(contributions.std(ddof=1)/2)
    assert policy["ci_lower_profit"] <= policy["dr_incremental_profit"] <= policy["ci_upper_profit"]


@pytest.mark.parametrize("change", ["decisions", "nan", "propensity", "length", "dimension"])
def test_invalid_evaluation_inputs_fail(change):
    args = dict(outcome=[1, 0], treatment=[1, 0], decisions=[1, 0],
                mu0=[0.4]*2, mu1=[0.6]*2, propensity=[0.5]*2)
    if change == "decisions": args["decisions"] = [0.5, 1]
    if change == "nan": args["mu0"] = [np.nan, 0.4]
    if change == "propensity": args["propensity"] = [1.1, 0.5]
    if change == "length": args["outcome"] = [1]
    if change == "dimension": args["outcome"] = [[1, 0]]
    with pytest.raises(ValueError):
        evaluate_observational_policy(**args)
