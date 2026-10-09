"""Validate interventions and distinguish causal from observational truth."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulation import simulate_customers, MODEL_FEATURES
from stress_simulation import simulate_stress_customers, STRESS_ORACLE_COLUMNS


def test_baseline_is_exactly_preserved():
    original = simulate_customers(2000, seed=42)
    stress = simulate_stress_customers(2000, seed=42)
    pd.testing.assert_frame_equal(stress[original.columns], original)
    np.testing.assert_allclose(stress.observational_mu0, stress.true_mu0)
    np.testing.assert_allclose(stress.observational_mu1, stress.true_mu1)


def test_overlap_changes_assignment_but_not_response_surfaces():
    baseline = simulate_stress_customers(10000, seed=42)
    severe = simulate_stress_customers(10000, seed=42, overlap_strength=5)
    pd.testing.assert_frame_equal(baseline[MODEL_FEATURES], severe[MODEL_FEATURES])
    np.testing.assert_allclose(baseline.true_mu0, severe.true_mu0)
    np.testing.assert_allclose(baseline.true_mu1, severe.true_mu1)
    weak = lambda frame: ((frame.true_propensity < 0.05) | (frame.true_propensity > 0.95)).mean()
    assert weak(severe) > weak(baseline)+0.2
    assert severe.true_propensity.between(0.001, 0.999).all()


def test_hidden_confounder_creates_assignment_selection_and_identification_bias():
    data = simulate_stress_customers(30000, seed=42, hidden_strength=1.6)
    assert data.groupby("hidden_confounder").treatment.mean().diff().iloc[-1] > 0.3
    observed_contrast = data.observational_mu1-data.observational_mu0
    assert (observed_contrast-data.true_cate).mean() > 0.1
    assert not set(STRESS_ORACLE_COLUMNS) & set(MODEL_FEATURES)
    # Causal truth marginalizes U, rather than comparing X-only predictions to U-specific effects.
    assert not np.allclose(data.true_mu0, data.latent_mu0)
    np.testing.assert_allclose(data.true_cate, data.true_mu1-data.true_mu0)


def test_deterministic_paired_features_and_latent_states():
    first = simulate_stress_customers(1000, seed=19, overlap_strength=5, hidden_strength=1.6)
    second = simulate_stress_customers(1000, seed=19, overlap_strength=5, hidden_strength=1.6)
    other = simulate_stress_customers(1000, seed=19, hidden_strength=0.8)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first[MODEL_FEATURES], other[MODEL_FEATURES])
    np.testing.assert_array_equal(first.hidden_confounder, other.hidden_confounder)
    assert first.true_cate.between(-1, 1).all()


@pytest.mark.parametrize("kwargs", [dict(overlap_strength=0.5), dict(hidden_strength=-1),
                                    dict(hidden_strength=np.nan), dict(overlap_strength=21)])
def test_invalid_interventions_fail(kwargs):
    with pytest.raises(ValueError): simulate_stress_customers(100, **kwargs)
