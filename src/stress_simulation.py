"""Controlled overlap and hidden-confounding interventions on the baseline DGP."""
from __future__ import annotations
import numpy as np
from simulation import simulate_customers, sigmoid

STRESS_ORACLE_COLUMNS = ["hidden_confounder", "latent_propensity", "latent_mu0", "latent_mu1",
                         "observational_mu0", "observational_mu1"]


def _logit(probability):
    return np.log(probability) - np.log1p(-probability)


def simulate_stress_customers(number_of_customers=20_000, seed=42, *,
                              overlap_strength=1.0, hidden_strength=0.0):
    """Preserve X, change assignment sharpness and a latent common cause.

    U is an independent Bernoulli(0.5) pre-treatment common cause, coded +/-1.
    hidden_strength shifts BOTH assignment and outcome logits. Published causal
    response surfaces integrate U out to target CATE conditional on observed X.
    Latent and observational surfaces are evaluation-only diagnostics.
    """
    if not np.isfinite([overlap_strength, hidden_strength]).all():
        raise ValueError("Stress strengths must be finite")
    if overlap_strength < 1 or hidden_strength < 0:
        raise ValueError("overlap_strength must be >= 1 and hidden_strength >= 0")
    if overlap_strength > 20 or hidden_strength > 10:
        raise ValueError("Stress strengths exceed the supported bounded range")
    base = simulate_customers(number_of_customers, seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 8]))
    u = rng.integers(0, 2, len(base))
    e = base.true_propensity.to_numpy()
    # Recover shared uniform draws consistent with the original observations.
    # At default strengths this exactly recreates baseline treatment/outcome.
    t0 = base.treatment.to_numpy()
    t_noise = rng.random(len(base))
    treatment_uniform = np.where(t0 == 1, t_noise*e, e+t_noise*(1-e))
    base_observed_mu = np.where(t0 == 1, base.true_mu1, base.true_mu0)
    y_noise = rng.random(len(base))
    outcome_uniform = np.where(base.retained_90d == 1, y_noise*base_observed_mu,
                               base_observed_mu+y_noise*(1-base_observed_mu))
    assignment = _logit(e)
    assignment = assignment.mean() + overlap_strength*(assignment-assignment.mean())
    mu0_logit = _logit(base.true_mu0.to_numpy())
    mu1_logit = _logit(base.true_mu1.to_numpy())
    # Column 0 is U=-1, column 1 is U=+1; integrate the two equiprobable states.
    offsets = hidden_strength*np.array([-1.0, 1.0])
    e_states = np.clip(sigmoid(assignment[:, None]+offsets), 0.001, 0.999)
    mu0_states = sigmoid(mu0_logit[:, None]+offsets)
    mu1_states = sigmoid(mu1_logit[:, None]+offsets)
    # Avoid altering baseline's clipped propensity by numerical roundoff.
    if overlap_strength == 1 and hidden_strength == 0:
        e_states = np.repeat(e[:, None], 2, axis=1)
        mu0_states = np.repeat(base.true_mu0.to_numpy()[:, None], 2, axis=1)
        mu1_states = np.repeat(base.true_mu1.to_numpy()[:, None], 2, axis=1)
    rows = np.arange(len(base))
    latent_e = e_states[rows, u]
    latent_mu0, latent_mu1 = mu0_states[rows, u], mu1_states[rows, u]
    treatment = (treatment_uniform < latent_e).astype(int)
    outcome = (outcome_uniform < np.where(treatment == 1, latent_mu1, latent_mu0)).astype(int)
    result = base.copy()
    result["treatment"] = treatment
    result["retained_90d"] = outcome
    result["true_propensity"] = e_states.mean(axis=1)
    result["true_mu0"] = mu0_states.mean(axis=1)
    result["true_mu1"] = mu1_states.mean(axis=1)
    result["true_cate"] = result.true_mu1-result.true_mu0
    result["hidden_confounder"] = u
    result["latent_propensity"] = latent_e
    result["latent_mu0"] = latent_mu0
    result["latent_mu1"] = latent_mu1
    # E[Y | X,T] differs from E[Y(T) | X] when assignment selects on latent U.
    result["observational_mu1"] = (mu1_states*e_states).sum(axis=1)/e_states.sum(axis=1)
    result["observational_mu0"] = (mu0_states*(1-e_states)).sum(axis=1)/(1-e_states).sum(axis=1)
    return result
