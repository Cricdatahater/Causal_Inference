import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from policies import (
    evaluate_policy,
    oracle_policy,
    policy_capacity,
    random_policy,
    top_score_policy,
    treat_all_policy,
    treat_none_policy,
)
from simulation import simulate_customers


def test_policy_capacity() -> None:
    assert policy_capacity(100, 0.20) == 20
    assert policy_capacity(103, 0.20) == 20


def test_treat_none_policy() -> None:
    decisions = treat_none_policy(100)

    assert len(decisions) == 100
    assert decisions.sum() == 0


def test_treat_all_policy() -> None:
    decisions = treat_all_policy(100)

    assert len(decisions) == 100
    assert decisions.sum() == 100


def test_random_policy_respects_capacity() -> None:
    decisions = random_policy(
        population_size=1_000,
        capacity=0.20,
        seed=42,
    )

    assert decisions.sum() == 200
    assert set(np.unique(decisions)) == {0, 1}


def test_random_policy_is_reproducible() -> None:
    first = random_policy(1_000, capacity=0.20, seed=42)
    second = random_policy(1_000, capacity=0.20, seed=42)

    np.testing.assert_array_equal(first, second)


def test_top_score_policy_selects_highest_scores() -> None:
    scores = np.array([0.10, 0.80, 0.40, 0.90, 0.20])

    decisions = top_score_policy(
        scores=scores,
        capacity=0.40,
    )

    expected = np.array([0, 1, 0, 1, 0])
    np.testing.assert_array_equal(decisions, expected)


def test_oracle_policy_respects_capacity() -> None:
    data = simulate_customers(
        number_of_customers=1_000,
        seed=42,
    )

    decisions = oracle_policy(
        data=data,
        capacity=0.20,
    )

    assert decisions.sum() <= 200


def test_treat_none_has_zero_incremental_profit() -> None:
    data = simulate_customers(
        number_of_customers=1_000,
        seed=42,
    )

    results = evaluate_policy(
        data=data,
        decisions=treat_none_policy(len(data)),
        policy_name="Treat nobody",
    )

    assert results["incremental_profit"] == 0
    assert results["treated_count"] == 0


def test_treat_all_is_not_capacity_feasible() -> None:
    data = simulate_customers(
        number_of_customers=1_000,
        seed=42,
    )

    results = evaluate_policy(
        data=data,
        decisions=treat_all_policy(len(data)),
        policy_name="Treat everybody",
        capacity=0.20,
    )

    assert results["capacity_feasible"] is False