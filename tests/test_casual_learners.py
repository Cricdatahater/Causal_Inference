import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from causal_learners import SLearner, TLearner
from simulation import simulate_customers


@pytest.fixture
def simulated_data():
    return simulate_customers(
        number_of_customers=3_000,
        seed=42,
    )


@pytest.mark.parametrize(
    "learner_class",
    [SLearner, TLearner],
)
def test_cate_prediction_shape(
    simulated_data,
    learner_class,
):
    learner = learner_class(
        random_seed=42
    )

    learner.fit(simulated_data)

    predictions = learner.predict_cate(
        simulated_data.iloc[:100]
    )

    assert predictions.shape == (100,)


@pytest.mark.parametrize(
    "learner_class",
    [SLearner, TLearner],
)
def test_cate_predictions_are_valid(
    simulated_data,
    learner_class,
):
    learner = learner_class(
        random_seed=42
    )

    learner.fit(simulated_data)

    predictions = learner.predict_cate(
        simulated_data.iloc[:100]
    )

    assert np.isfinite(predictions).all()
    assert (predictions >= -1).all()
    assert (predictions <= 1).all()


@pytest.mark.parametrize(
    "learner_class",
    [SLearner, TLearner],
)
def test_potential_outcomes_are_probabilities(
    simulated_data,
    learner_class,
):
    learner = learner_class(
        random_seed=42
    )

    learner.fit(simulated_data)

    predicted_mu0, predicted_mu1 = (
        learner.predict_potential_outcomes(
            simulated_data.iloc[:100]
        )
    )

    assert predicted_mu0.shape == (100,)
    assert predicted_mu1.shape == (100,)

    assert (predicted_mu0 >= 0).all()
    assert (predicted_mu0 <= 1).all()
    assert (predicted_mu1 >= 0).all()
    assert (predicted_mu1 <= 1).all()


@pytest.mark.parametrize(
    "learner_class",
    [SLearner, TLearner],
)
def test_predictions_are_reproducible(
    simulated_data,
    learner_class,
):
    first = learner_class(
        random_seed=42
    ).fit(simulated_data)

    second = learner_class(
        random_seed=42
    ).fit(simulated_data)

    first_predictions = first.predict_cate(
        simulated_data.iloc[:100]
    )

    second_predictions = second.predict_cate(
        simulated_data.iloc[:100]
    )

    np.testing.assert_allclose(
        first_predictions,
        second_predictions,
    )


@pytest.mark.parametrize(
    "learner_class",
    [SLearner, TLearner],
)
def test_prediction_before_fit_fails(
    simulated_data,
    learner_class,
):
    learner = learner_class()

    with pytest.raises(RuntimeError):
        learner.predict_cate(
            simulated_data.iloc[:10]
        )