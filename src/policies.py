"""Treatment policies and policy-value evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def policy_capacity(
    population_size: int,
    capacity: float,
) -> int:
    """Return the maximum number of customers that may be treated."""
    if population_size <= 0:
        raise ValueError("population_size must be positive")

    if not 0 <= capacity <= 1:
        raise ValueError("capacity must be between 0 and 1")

    return int(np.floor(population_size * capacity))


def treat_none_policy(population_size: int) -> np.ndarray:
    """Treat no customers."""
    if population_size <= 0:
        raise ValueError("population_size must be positive")

    return np.zeros(population_size, dtype=int)


def treat_all_policy(population_size: int) -> np.ndarray:
    """Treat every customer."""
    if population_size <= 0:
        raise ValueError("population_size must be positive")

    return np.ones(population_size, dtype=int)


def random_policy(
    population_size: int,
    capacity: float = 0.20,
    seed: int = 42,
) -> np.ndarray:
    """Randomly treat exactly the permitted number of customers."""
    maximum_treated = policy_capacity(population_size, capacity)
    rng = np.random.default_rng(seed)

    selected_indices = rng.choice(
        population_size,
        size=maximum_treated,
        replace=False,
    )

    decisions = np.zeros(population_size, dtype=int)
    decisions[selected_indices] = 1

    return decisions


def top_score_policy(
    scores: np.ndarray | pd.Series,
    capacity: float = 0.20,
    require_positive: bool = False,
) -> np.ndarray:
    """Treat customers with the highest supplied decision scores.

    Parameters
    ----------
    scores:
        Larger values indicate a greater reason to treat.
    capacity:
        Maximum share of customers who may be treated.
    require_positive:
        If true, customers with non-positive scores are never treated.
    """
    score_array = np.asarray(scores, dtype=float)

    if score_array.ndim != 1:
        raise ValueError("scores must be one-dimensional")

    if len(score_array) == 0:
        raise ValueError("scores cannot be empty")

    if not np.isfinite(score_array).all():
        raise ValueError("scores must contain only finite values")

    maximum_treated = policy_capacity(
        population_size=len(score_array),
        capacity=capacity,
    )

    ranked_indices = np.argsort(-score_array, kind="stable")

    if require_positive:
        ranked_indices = ranked_indices[
            score_array[ranked_indices] > 0
        ]

    selected_indices = ranked_indices[:maximum_treated]

    decisions = np.zeros(len(score_array), dtype=int)
    decisions[selected_indices] = 1

    return decisions


def oracle_policy(
    data: pd.DataFrame,
    capacity: float = 0.20,
    retention_value: float = 80.0,
    treatment_cost: float = 10.0,
) -> np.ndarray:
    """Select customers with the greatest true incremental value."""
    required_columns = {"true_cate"}

    if not required_columns.issubset(data.columns):
        raise ValueError("data must contain true_cate")

    true_incremental_value = (
        retention_value * data["true_cate"] - treatment_cost
    )

    return top_score_policy(
        scores=true_incremental_value,
        capacity=capacity,
        require_positive=True,
    )


def evaluate_policy(
    data: pd.DataFrame,
    decisions: np.ndarray | pd.Series,
    policy_name: str,
    capacity: float = 0.20,
    retention_value: float = 80.0,
    treatment_cost: float = 10.0,
) -> dict[str, float | int | str | bool]:
    """Evaluate a policy using the simulator's true response surfaces."""
    required_columns = {
        "true_mu0",
        "true_mu1",
        "true_cate",
    }

    if not required_columns.issubset(data.columns):
        missing = required_columns.difference(data.columns)
        raise ValueError(f"Missing oracle columns: {sorted(missing)}")

    decision_array = np.asarray(decisions, dtype=int)

    if len(decision_array) != len(data):
        raise ValueError(
            "decisions must have the same number of rows as data"
        )

    if not set(np.unique(decision_array)).issubset({0, 1}):
        raise ValueError("decisions must contain only 0 and 1")

    treated_count = int(decision_array.sum())
    treatment_rate = float(decision_array.mean())

    individual_incremental_retention = (
        decision_array * data["true_cate"].to_numpy()
    )

    incremental_retained_customers = float(
        individual_incremental_retention.sum()
    )

    campaign_cost = float(treated_count * treatment_cost)

    incremental_revenue = float(
        incremental_retained_customers * retention_value
    )

    incremental_profit = incremental_revenue - campaign_cost

    expected_retention_probability = (
        data["true_mu0"].to_numpy()
        + individual_incremental_retention
    )

    mean_cate_treated = (
        float(
            data.loc[
                decision_array == 1,
                "true_cate",
            ].mean()
        )
        if treated_count > 0
        else 0.0
    )

    return {
        "policy": policy_name,
        "treated_count": treated_count,
        "treatment_rate": treatment_rate,
        "capacity_feasible": treatment_rate <= capacity + 1e-12,
        "mean_cate_treated": mean_cate_treated,
        "incremental_retained_customers": (
            incremental_retained_customers
        ),
        "expected_retention_rate": float(
            expected_retention_probability.mean()
        ),
        "incremental_revenue": incremental_revenue,
        "campaign_cost": campaign_cost,
        "incremental_profit": incremental_profit,
        "profit_per_treated_customer": (
            incremental_profit / treated_count
            if treated_count > 0
            else 0.0
        ),
    }