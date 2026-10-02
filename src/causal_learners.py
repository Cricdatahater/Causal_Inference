"""Causal meta-learners for heterogeneous treatment-effect estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_price",
    "login_days_30d",
    "usage_trend_90d",
    "support_tickets_30d",
    "late_payments_12m",
]

CATEGORICAL_FEATURES = [
    "plan_type",
    "region",
    "device_type",
]

MODEL_FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)

FORBIDDEN_MODEL_COLUMNS = {
    "customer_id",
    "retained_90d",
    "true_propensity",
    "true_mu0",
    "true_mu1",
    "true_cate",
}


def _validate_training_data(data: pd.DataFrame) -> None:
    """Validate the columns required by the causal learners."""
    required_columns = {
        *MODEL_FEATURES,
        "treatment",
        "retained_90d",
    }

    missing_columns = required_columns.difference(
        data.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if not set(
        data["treatment"].unique()
    ).issubset({0, 1}):
        raise ValueError(
            "treatment must contain only 0 and 1"
        )

    if not set(
        data["retained_90d"].unique()
    ).issubset({0, 1}):
        raise ValueError(
            "retained_90d must contain only 0 and 1"
        )


def _make_outcome_model(
    numeric_features: list[str],
    random_seed: int,
) -> Pipeline:
    """Construct a probabilistic outcome-model pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=25,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_seed,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


class SLearner:
    """Single-model treatment-effect estimator."""

    def __init__(
        self,
        random_seed: int = 42,
    ) -> None:
        self.random_seed = random_seed
        self.model: Pipeline | None = None

    def fit(
        self,
        data: pd.DataFrame,
    ) -> "SLearner":
        """Fit one model using covariates and treatment."""
        _validate_training_data(data)

        model_features = data[
            MODEL_FEATURES
        ].copy()

        model_features["treatment"] = (
            data["treatment"].to_numpy()
        )

        self.model = _make_outcome_model(
            numeric_features=(
                NUMERIC_FEATURES
                + ["treatment"]
            ),
            random_seed=self.random_seed,
        )

        self.model.fit(
            model_features,
            data["retained_90d"],
        )

        return self

    def predict_potential_outcomes(
        self,
        data: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict retention under control and treatment."""
        if self.model is None:
            raise RuntimeError(
                "The S-learner must be fitted first"
            )

        control_features = data[
            MODEL_FEATURES
        ].copy()

        treated_features = data[
            MODEL_FEATURES
        ].copy()

        control_features["treatment"] = 0
        treated_features["treatment"] = 1

        predicted_mu0 = self.model.predict_proba(
            control_features
        )[:, 1]

        predicted_mu1 = self.model.predict_proba(
            treated_features
        )[:, 1]

        return predicted_mu0, predicted_mu1

    def predict_cate(
        self,
        data: pd.DataFrame,
    ) -> np.ndarray:
        """Predict conditional average treatment effects."""
        predicted_mu0, predicted_mu1 = (
            self.predict_potential_outcomes(data)
        )

        return np.clip(
            predicted_mu1 - predicted_mu0,
            -1,
            1,
        )


class TLearner:
    """Separate-model treatment-effect estimator."""

    def __init__(
        self,
        random_seed: int = 42,
    ) -> None:
        self.random_seed = random_seed
        self.control_model: Pipeline | None = None
        self.treated_model: Pipeline | None = None

    def fit(
        self,
        data: pd.DataFrame,
    ) -> "TLearner":
        """Fit separate outcome models for each treatment arm."""
        _validate_training_data(data)

        control_data = data.loc[
            data["treatment"] == 0
        ]

        treated_data = data.loc[
            data["treatment"] == 1
        ]

        if control_data.empty or treated_data.empty:
            raise ValueError(
                "Both treatment groups are required"
            )

        self.control_model = _make_outcome_model(
            numeric_features=NUMERIC_FEATURES,
            random_seed=self.random_seed,
        )

        self.treated_model = _make_outcome_model(
            numeric_features=NUMERIC_FEATURES,
            random_seed=self.random_seed + 1,
        )

        self.control_model.fit(
            control_data[MODEL_FEATURES],
            control_data["retained_90d"],
        )

        self.treated_model.fit(
            treated_data[MODEL_FEATURES],
            treated_data["retained_90d"],
        )

        return self

    def predict_potential_outcomes(
        self,
        data: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict retention under control and treatment."""
        if (
            self.control_model is None
            or self.treated_model is None
        ):
            raise RuntimeError(
                "The T-learner must be fitted first"
            )

        features = data[MODEL_FEATURES]

        predicted_mu0 = (
            self.control_model.predict_proba(
                features
            )[:, 1]
        )

        predicted_mu1 = (
            self.treated_model.predict_proba(
                features
            )[:, 1]
        )

        return predicted_mu0, predicted_mu1

    def predict_cate(
        self,
        data: pd.DataFrame,
    ) -> np.ndarray:
        """Predict conditional average treatment effects."""
        predicted_mu0, predicted_mu1 = (
            self.predict_potential_outcomes(data)
        )

        return np.clip(
            predicted_mu1 - predicted_mu0,
            -1,
            1,
        )