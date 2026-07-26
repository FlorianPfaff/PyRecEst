import pytest
from pyrecest.models import (
    LinearGaussianMeasurementModel,
    LinearGaussianTransitionModel,
)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LinearGaussianTransitionModel([[True]], [[1.0]]),
        lambda: LinearGaussianTransitionModel([[1.0]], [[True]]),
        lambda: LinearGaussianTransitionModel([[1.0]], [[1.0]], offset=[True]),
        lambda: LinearGaussianMeasurementModel([[True]], [[1.0]]),
        lambda: LinearGaussianMeasurementModel([[1.0]], [[True]]),
    ],
)
def test_linear_gaussian_constructors_reject_boolean_parameters(factory) -> None:
    with pytest.raises(ValueError, match="real numeric"):
        factory()


def test_transition_model_rejects_boolean_state_inputs() -> None:
    model = LinearGaussianTransitionModel([[1.0]], [[1.0]])

    with pytest.raises(ValueError, match="real numeric"):
        model.predict_mean([True])
    with pytest.raises(ValueError, match="real numeric"):
        model.predict_covariance([[True]])


def test_measurement_model_rejects_boolean_state_inputs() -> None:
    model = LinearGaussianMeasurementModel([[1.0]], [[1.0]])

    with pytest.raises(ValueError, match="real numeric"):
        model.predict_mean([True])
    with pytest.raises(ValueError, match="real numeric"):
        model.innovation_covariance([[True]])
