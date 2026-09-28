import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from simulation import ORACLE_COLUMNS, simulate_customers  # noqa: E402


def test_simulation_is_reproducible() -> None:
    first = simulate_customers(500, seed=10)
    second = simulate_customers(500, seed=10)

    pd.testing.assert_frame_equal(first, second)


def test_simulation_has_expected_schema() -> None:
    data = simulate_customers(500, seed=11)

    required_columns = {
        "customer_id",
        "tenure_months",
        "plan_type",
        "monthly_price",
        "login_days_30d",
        "usage_trend_90d",
        "support_tickets_30d",
        "late_payments_12m",
        "region",
        "device_type",
        "treatment",
        "retained_90d",
        *ORACLE_COLUMNS,
    }

    assert required_columns.issubset(data.columns)


def test_probabilities_and_outcomes_are_valid() -> None:
    data = simulate_customers(2_000, seed=12)

    assert data["true_propensity"].between(0.05, 0.80).all()
    assert data["true_mu0"].between(0, 1).all()
    assert data["true_mu1"].between(0, 1).all()
    assert set(data["treatment"].unique()).issubset({0, 1})
    assert set(data["retained_90d"].unique()).issubset({0, 1})


def test_treatment_effects_are_heterogeneous() -> None:
    data = simulate_customers(5_000, seed=13)

    assert data["true_cate"].std() > 0.02
    assert data["true_cate"].nunique() > 100


def test_both_treatment_groups_are_represented() -> None:
    data = simulate_customers(2_000, seed=14)
    treatment_rate = data["treatment"].mean()

    assert 0.20 < treatment_rate < 0.45