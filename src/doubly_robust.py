"""Cross-fitted doubly robust CATE learning and observational policy evaluation."""
from __future__ import annotations

from statistics import NormalDist
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from causal_learners import MODEL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def _vectors(**arrays):
    result = {name: np.asarray(value, dtype=float) for name, value in arrays.items()}
    sizes = {len(value) for value in result.values() if value.ndim == 1}
    if len(sizes) != 1 or not sizes or next(iter(sizes)) == 0:
        raise ValueError("Inputs must be nonempty vectors with matching lengths")
    if any(value.ndim != 1 or not np.isfinite(value).all() for value in result.values()):
        raise ValueError("Inputs must be finite one-dimensional vectors")
    return result


def _binary(name, value):
    if not np.isin(value, [0, 1]).all():
        raise ValueError(f"{name} must contain only 0 and 1")


def _clip_propensity(propensity, clip):
    if not np.isfinite(clip) or not 0 < clip < 0.5:
        raise ValueError("propensity_clip must lie strictly between 0 and 0.5")
    if ((propensity < 0) | (propensity > 1)).any():
        raise ValueError("propensity must be a probability")
    return np.clip(propensity, clip, 1 - clip)


def doubly_robust_scores(outcome, treatment, mu0, mu1, propensity, *, propensity_clip=0.05):
    """Return unbounded AIPW scores for Y(0), Y(1), and their difference.

    Nuisance estimates must be out of sample. Neither observed outcomes nor
    pseudo-outcomes are clipped: clipping them would change the estimand.
    """
    arrays = _vectors(outcome=outcome, treatment=treatment, mu0=mu0,
                      mu1=mu1, propensity=propensity)
    y, t, m0, m1, e = (arrays[key] for key in arrays)
    _binary("outcome", y)
    _binary("treatment", t)
    if ((m0 < 0) | (m0 > 1) | (m1 < 0) | (m1 > 1)).any():
        raise ValueError("Outcome nuisance estimates must be probabilities")
    e = _clip_propensity(e, propensity_clip)
    score0 = m0 + (1 - t) * (y - m0) / (1 - e)
    score1 = m1 + t * (y - m1) / e
    return score0, score1, score1 - score0


def _pipeline(estimator):
    # Always select the explicit pre-treatment whitelist; ignore oracle columns.
    preprocessing = ColumnTransformer([
        ("numeric", "passthrough", NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocessor", preprocessing), ("model", estimator)])


def _probability(model, features):
    """Handle one-class outcome folds without assuming a second probability column."""
    classes = model.named_steps["model"].classes_
    indices = np.flatnonzero(classes == 1)
    return (model.predict_proba(features)[:, indices[0]] if len(indices)
            else np.zeros(len(features)))


class DRLearner:
    """Regress cross-fitted AIPW effect scores on pre-treatment covariates.

    Propensity/outcome models are unweighted probability forests. Fold models
    are retained and averaged for out-of-sample nuisance predictions. Final CATE
    predictions are clipped to the binary-outcome range, while scores stay raw.
    """
    def __init__(self, random_seed=42, n_splits=3, propensity_clip=0.05,
                 n_estimators=200, min_samples_leaf=25):
        if not isinstance(n_splits, int) or n_splits < 2:
            raise ValueError("n_splits must be an integer of at least 2")
        _clip_propensity(np.array([0.5]), propensity_clip)
        if not isinstance(n_estimators, int) or n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer")
        if not isinstance(min_samples_leaf, int) or min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be a positive integer")
        self.random_seed = random_seed
        self.n_splits = n_splits
        self.propensity_clip = propensity_clip
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.final_model_ = None

    def _classifier(self, seed):
        return _pipeline(RandomForestClassifier(
            n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf,
            max_features=1.0, random_state=seed, n_jobs=-1))

    def fit(self, data):
        # Reset fitted state so a failed refit cannot silently serve old predictions.
        self.final_model_ = None
        missing = set(MODEL_FEATURES + ["treatment", "retained_90d"]) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        x = data[MODEL_FEATURES]
        y = data["retained_90d"].to_numpy(dtype=float)
        t = data["treatment"].to_numpy(dtype=float)
        _vectors(outcome=y, treatment=t)
        _binary("outcome", y)
        _binary("treatment", t)
        if any(np.count_nonzero(t == arm) < self.n_splits for arm in (0, 1)):
            raise ValueError("Each treatment arm needs at least n_splits observations")
        self.fold_models_ = []
        self.fold_indices_ = []
        self.fold_assignments_ = np.full(len(data), -1, dtype=int)
        e, m0, m1 = (np.empty(len(data)) for _ in range(3))
        splitter = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_seed)
        for fold, (fit_indices, heldout_indices) in enumerate(splitter.split(x, t)):
            seed = self.random_seed + 3 * fold
            propensity_model = self._classifier(seed)
            control_model = self._classifier(seed + 1)
            treated_model = self._classifier(seed + 2)
            propensity_model.fit(x.iloc[fit_indices], t[fit_indices])
            control_indices = fit_indices[t[fit_indices] == 0]
            treated_indices = fit_indices[t[fit_indices] == 1]
            control_model.fit(x.iloc[control_indices], y[control_indices])
            treated_model.fit(x.iloc[treated_indices], y[treated_indices])
            heldout = x.iloc[heldout_indices]
            e[heldout_indices] = _probability(propensity_model, heldout)
            m0[heldout_indices] = _probability(control_model, heldout)
            m1[heldout_indices] = _probability(treated_model, heldout)
            self.fold_assignments_[heldout_indices] = fold
            self.fold_indices_.append((fit_indices.copy(), heldout_indices.copy()))
            self.fold_models_.append((propensity_model, control_model, treated_model))
        self.oof_nuisance_ = pd.DataFrame({"propensity": e, "mu0": m0, "mu1": m1})
        _, _, self.pseudo_outcome_ = doubly_robust_scores(
            y, t, m0, m1, e, propensity_clip=self.propensity_clip)
        self.final_model_ = _pipeline(RandomForestRegressor(
            n_estimators=self.n_estimators, min_samples_leaf=max(50, self.min_samples_leaf),
            max_features=1.0, random_state=self.random_seed, n_jobs=-1))
        self.final_model_.fit(x, self.pseudo_outcome_)
        return self

    def _require_fit(self):
        if self.final_model_ is None:
            raise RuntimeError("The DR-learner must be fitted first")

    def predict_cate(self, data):
        self._require_fit()
        return np.clip(self.final_model_.predict(data[MODEL_FEATURES]), -1, 1)

    def predict_nuisance(self, data):
        """Return raw propensity and outcome probabilities for NEW observations.

        For training observations use oof_nuisance_, never this ensemble method.
        """
        self._require_fit()
        x = data[MODEL_FEATURES]
        predictions = np.array([
            [_probability(model, x) for model in models]
            for models in self.fold_models_
        ]).mean(axis=0)
        return pd.DataFrame(dict(zip(["propensity", "mu0", "mu1"], predictions)))


def evaluate_observational_policy(outcome, treatment, decisions, mu0, mu1, propensity,
                                  *, policy_name="Policy", retention_value=80.0,
                                  treatment_cost=10.0, propensity_clip=0.05,
                                  confidence_level=0.95):
    """Estimate incremental profit vs treat-none using paired AIPW scores.

    All models/policy thresholds must be fitted without evaluation outcomes.
    Normal intervals are pointwise approximations for a fixed policy under iid
    sampling and suitable nuisance convergence. A cohort-wide capacity cap adds
    dependence; these intervals do not certify that deployment rule or account
    for policy selection, nuisance fitting, or multiple comparisons.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    if not np.isfinite([retention_value, treatment_cost]).all() or min(retention_value, treatment_cost) < 0:
        raise ValueError("Financial values must be finite and nonnegative")
    arrays = _vectors(outcome=outcome, treatment=treatment, decisions=decisions,
                      mu0=mu0, mu1=mu1, propensity=propensity)
    d = arrays["decisions"]
    _binary("decisions", d)
    if len(d) < 2:
        raise ValueError("At least two evaluation observations are required")
    s0, s1, tau = doubly_robust_scores(
        arrays["outcome"], arrays["treatment"], arrays["mu0"], arrays["mu1"],
        arrays["propensity"], propensity_clip=propensity_clip)
    contributions = d * (retention_value * tau - treatment_cost)
    average = float(contributions.mean())
    se = float(contributions.std(ddof=1) / np.sqrt(len(d)))
    z = NormalDist().inv_cdf((1 + confidence_level) / 2)
    raw_e = arrays["propensity"]
    clipped_e = _clip_propensity(raw_e, propensity_clip)
    weights = np.where(arrays["treatment"] == 1, 1 / clipped_e, 1 / (1 - clipped_e))
    return {
        "policy": policy_name, "evaluation_customers": len(d),
        "treated_count": int(d.sum()), "treatment_rate": float(d.mean()),
        "dr_retention_rate": float((s0 + d * (s1 - s0)).mean()),
        "dr_profit_per_customer": average, "standard_error_per_customer": se,
        "confidence_level": confidence_level,
        "ci_lower_per_customer": average - z * se,
        "ci_upper_per_customer": average + z * se,
        "dr_incremental_profit": len(d) * average,
        "ci_lower_profit": len(d) * (average - z * se),
        "ci_upper_profit": len(d) * (average + z * se),
        "propensity_clipped_fraction": float((raw_e != clipped_e).mean()),
        "maximum_observed_inverse_weight": float(weights.max()),
        "observed_weight_ess": float(weights.sum() ** 2 / np.square(weights).sum()),
    }
