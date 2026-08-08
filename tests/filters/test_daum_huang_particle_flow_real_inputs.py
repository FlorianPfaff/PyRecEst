"""Regression tests for real-valued Daum-Huang particle-flow inputs."""

import numpy as np
import pytest

from pyrecest.backend import array
from pyrecest.filters.daum_huang_particle_filter import (
    gaussian_bridge_moments,
    gaussian_particle_flow_update,
)
from pyrecest.models import LinearGaussianMeasurementModel


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    (
        ("mean", array([1.0 + 2.0j]), "mean"),
        ("covariance", array([[1.0 + 0.0j]]), "covariance"),
        ("measurement_matrix", array([[1.0 + 2.0j]]), "measurement_matrix"),
        ("measurement", array([1.0 + 2.0j]), "measurement"),
        (
            "measurement_noise_covariance",
            array([[1.0 + 0.0j]]),
            "measurement_noise_covariance",
        ),
    ),
)
def test_gaussian_bridge_rejects_complex_public_inputs(argument, value, message):
    kwargs = {
        "mean": array([0.0]),
        "covariance": array([[1.0]]),
        "measurement_matrix": array([[1.0]]),
        "measurement": array([0.0]),
        "measurement_noise_covariance": array([[1.0]]),
        "delta_lambda": 1.0,
    }
    kwargs[argument] = value

    with pytest.raises(ValueError, match=rf"{message}.*real"):
        gaussian_bridge_moments(**kwargs)


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    (
        ("particles", array([[-1.0 + 2.0j], [1.0 + 0.0j]]), "particles"),
        ("weights", array([0.5 + 0.0j, 0.5 + 1.0j]), "weights"),
        (
            "step_schedule",
            array([0.5 + 1.0j, 0.5 + 0.0j]),
            "step_schedule",
        ),
    ),
)
def test_particle_flow_rejects_complex_state_and_control_inputs(
    argument, value, message
):
    model = LinearGaussianMeasurementModel(array([[1.0]]), array([[1.0]]))
    kwargs = {
        "particles": array([[-1.0], [1.0]]),
        "measurement_model": model,
        "measurement": array([0.0]),
    }
    kwargs[argument] = value

    with pytest.raises(ValueError, match=rf"{message}.*real"):
        gaussian_particle_flow_update(**kwargs)


class _ComplexBatchMeasurementModel:
    noise_covariance = array([[1.0]])

    @staticmethod
    def h(states):
        states = np.asarray(states)
        return states[..., :1].astype(complex) + 1.0j

    @staticmethod
    def jacobian(states):
        states = np.asarray(states)
        return np.ones(states.shape[:-1] + (1, 1), dtype=float)


def test_particle_flow_rejects_complex_vectorized_measurement_output():
    with pytest.raises(ValueError, match=r"measurement value.*real"):
        gaussian_particle_flow_update(
            array([[-1.0], [1.0]]),
            _ComplexBatchMeasurementModel(),
            array([0.0]),
        )
