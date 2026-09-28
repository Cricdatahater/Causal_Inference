"""Synthetic observational data for causal retention modelling."""

from __future__ import annotations

import numpy as np
import pandas as pd


MODEL_FEATURES = [
    "tenure_months",
    "plan_type",
    "monthly_price",
    "login_days_30d",
    "usage_trend_90d",
    "support_tickets_30d",
    "late_payments_12m",
    "region",
    "device_type",
]

ORACLE_COLUMNS = [
    "true_propensity",
    "true_mu0",
    "true_mu1",
    "true_cate",
]


def sigmoid(value: np.ndarray) -> np.ndarray:
    """Numerically stable enough for the bounded logits used here."""
    return 1.0 / (1.0 + np.exp(-value))


def standardize(value: np.ndarray) -> np.ndarray:
    """Standardize a generated variable within the simulated population."""
    standard_deviation = value.std()

    if standard_deviation == 0:
        return np.zeros_like(value, dtype=float)

    return (value - value.mean()) / standard_deviation


def simulate_customers(
    number_of_customers: int = 20_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate one observational retention-campaign cohort.

    The treatment assignment and outcome mechanisms depend only on observed
    pre-treatment variables. Conditional exchangeability therefore holds by
    construction in this baseline simulation.
    """
    if number_of_customers <= 0:
        raise ValueError("number_of_customers must be positive")

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------
    # 1. Generate customer characteristics
    # ---------------------------------------------------------------

    tenure_months = np.clip(
        rng.gamma(shape=2.2, scale=16.0, size=number_of_customers),
        1,
        120,
    ).round().astype(int)

    plan_score = (
        0.025 * tenure_months
        + rng.normal(0, 1, number_of_customers)
    )

    plan_type = np.select(
        [plan_score < 0.40, plan_score < 1.60],
        ["basic", "standard"],
        default="premium",
    )

    base_price = pd.Series(plan_type).map(
        {
            "basic": 25.0,
            "standard": 50.0,
            "premium": 85.0,
        }
    ).to_numpy()

    monthly_price = np.clip(
        base_price + rng.normal(0, 4, number_of_customers),
        10,
        120,
    ).round(2)

    region = rng.choice(
        ["north", "south", "east", "west"],
        size=number_of_customers,
        p=[0.25, 0.30, 0.25, 0.20],
    )

    device_type = rng.choice(
        ["mobile", "desktop", "tablet"],
        size=number_of_customers,
        p=[0.60, 0.30, 0.10],
    )

    engagement_logit = (
        -0.45
        + 0.018 * tenure_months
        + 0.30 * (plan_type == "standard")
        + 0.50 * (plan_type == "premium")
        - 0.15 * (device_type == "tablet")
    )

    daily_login_probability = np.clip(
        sigmoid(engagement_logit),
        0.03,
        0.95,
    )

    login_days_30d = rng.binomial(
        n=30,
        p=daily_login_probability,
    )

    usage_trend_90d = np.clip(
        rng.normal(
            loc=0.018 * (login_days_30d - 15),
            scale=0.35,
            size=number_of_customers,
        ),
        -1,
        1,
    ).round(3)

    support_rate = np.exp(
        -0.35
        - 0.75 * usage_trend_90d
        + 0.25 * (login_days_30d < 8)
    )

    support_tickets_30d = np.clip(
        rng.poisson(support_rate),
        0,
        10,
    )

    late_payment_rate = np.exp(
        -1.25
        + 0.40 * (plan_type == "basic")
        + 0.20 * (region == "south")
    )

    late_payments_12m = np.clip(
        rng.poisson(late_payment_rate),
        0,
        12,
    )

    # Standardized representations used only inside the DGP.
    tenure_z = standardize(tenure_months)
    price_z = standardize(monthly_price)
    login_z = standardize(login_days_30d)
    support_z = standardize(support_tickets_30d)
    late_payment_z = standardize(late_payments_12m)

    # ---------------------------------------------------------------
    # 2. Baseline retention without treatment
    # ---------------------------------------------------------------

    untreated_retention_logit = (
        0.55
        + 0.35 * tenure_z
        + 0.55 * login_z
        + 0.70 * usage_trend_90d
        - 0.35 * support_z
        - 0.30 * late_payment_z
        + 0.15 * (plan_type == "premium")
        - 0.10 * (region == "south")
    )

    true_mu0 = sigmoid(untreated_retention_logit)

    # ---------------------------------------------------------------
    # 3. Heterogeneous treatment response
    # ---------------------------------------------------------------

    # The Gaussian-shaped term creates a "persuadable middle":
    # customers with moderately negative usage trends receive the
    # largest treatment benefit.
    moderate_disengagement = np.exp(
        -((usage_trend_90d + 0.25) / 0.45) ** 2
    )

    treatment_log_odds_effect = (
        0.15
        + 1.10 * moderate_disengagement
        + 0.25 * (plan_type == "basic")
        - 0.20 * (plan_type == "premium")
        - 0.30 * (support_tickets_30d >= 3)
        + 0.10 * (device_type == "mobile")
    )

    true_mu1 = sigmoid(
        untreated_retention_logit + treatment_log_odds_effect
    )

    true_cate = true_mu1 - true_mu0

    # ---------------------------------------------------------------
    # 4. Confounded historical treatment assignment
    # ---------------------------------------------------------------

    treatment_assignment_logit = (
        -0.95
        - 0.45 * login_z
        - 0.65 * usage_trend_90d
        + 0.40 * support_z
        + 0.25 * late_payment_z
        + 0.15 * price_z
        + 0.15 * (plan_type == "premium")
    )

    true_propensity = np.clip(
        sigmoid(treatment_assignment_logit),
        0.05,
        0.80,
    )

    treatment = rng.binomial(
        n=1,
        p=true_propensity,
    )

    # ---------------------------------------------------------------
    # 5. Generate the observed outcome
    # ---------------------------------------------------------------

    observed_retention_probability = np.where(
        treatment == 1,
        true_mu1,
        true_mu0,
    )

    retained_90d = rng.binomial(
        n=1,
        p=observed_retention_probability,
    )

    return pd.DataFrame(
        {
            "customer_id": np.arange(1, number_of_customers + 1),
            "tenure_months": tenure_months,
            "plan_type": plan_type,
            "monthly_price": monthly_price,
            "login_days_30d": login_days_30d,
            "usage_trend_90d": usage_trend_90d,
            "support_tickets_30d": support_tickets_30d,
            "late_payments_12m": late_payments_12m,
            "region": region,
            "device_type": device_type,
            "treatment": treatment,
            "retained_90d": retained_90d,
            "true_propensity": true_propensity,
            "true_mu0": true_mu0,
            "true_mu1": true_mu1,
            "true_cate": true_cate,
        }
    )