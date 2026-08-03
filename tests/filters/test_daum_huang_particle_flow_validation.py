import numpy as np
import pytest
from pyrecest.backend import array, to_numpy
from pyrecest.filters.daum_huang_particle_filter import (
    gaussian_bridge_moments,
    gaussian_particle_flow_update,
)
from pyrecest.models import LinearGaussianMeasurementModel


def test_gaussian_bridge_rejects_negative_definite_state_covariance():
    with pytest.raises(ValueError, match="positive semidefinite"):
        gaussian_bridge_moments(
            array([0.0, 0.0]),
            array([[-1.0, 0.0], [0.0, -2.0]]),
            array([[1.0, 0.0]]),
            array([0.0]),
            array([[1.0]]),
            1.0,
        )


def test_gaussian_bridge_rejects_negative_definite_measurement_covariance():
    with pytest.raises(ValueError, match="positive semidefinite"):
        gaussian_bridge_moments(
            array([0.0, 0.0]),
            array([[1.0, 0.0], [0.0, 1.0]]),
            array([[1.0, 0.0], [0.0, 1.0]]),
            array([0.0, 0.0]),
            array([[-1.0, 0.0], [0.0, -2.0]]),
            1.0,
        )


def test_gaussian_bridge_regularizes_singular_positive_semidefinite_covariance():
    mean, covariance = gaussian_bridge_moments(
        array([0.0, 0.0]),
        array([[1.0, 0.0], [0.0, 0.0]]),
        array([[1.0, 0.0]]),
        array([0.0]),
        array([[1.0]]),
        1.0,
        jitter=0.0,
    )

    assert np.all(np.isfinite(to_numpy(mean)))
    assert np.all(np.linalg.eigvalsh(to_numpy(covariance)) > 0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"n_steps": np.ma.array(2, mask=True)}, "n_steps"),
        ({"jitter": np.ma.array(1e-8, mask=True)}, "jitter"),
        (
            {"step_schedule": np.ma.array([0.5, 0.5], mask=[False, True])},
            "step_schedule",
        ),
        ({"weights": np.ma.array([0.5, 0.5], mask=[False, True])}, "weights"),
    ),
)
def test_particle_flow_rejects_masked_controls(kwargs, message):
    model = LinearGaussianMeasurementModel(array([[1.0]]), array([[1.0]]))
    with pytest.raises(ValueError, match=message):
        gaussian_particle_flow_update(
            array([[-1.0], [1.0]]),
            model,
            array([0.0]),
            **kwargs,
        )


def test_particle_flow_rejects_masked_particles_and_measurements():
    model = LinearGaussianMeasurementModel(array([[1.0]]), array([[1.0]]))

    with pytest.raises(ValueError, match="particles"):
        gaussian_particle_flow_update(
            np.ma.array([[-1.0], [1.0]], mask=[[False], [True]]),
            model,
            array([0.0]),
        )

    with pytest.raises(ValueError, match="measurement"):
        gaussian_particle_flow_update(
            array([[-1.0], [1.0]]),
            model,
            np.ma.array([0.0], mask=[True]),
        )
