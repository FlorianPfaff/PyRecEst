"""Regression tests for scale-stable Daum-Huang normalization."""

import numpy as np
import numpy.testing as npt
import pytest

from pyrecest import backend
from pyrecest.filters.daum_huang_particle_filter import gaussian_particle_flow_update

pytestmark = pytest.mark.skipif(
    backend.__backend_name__ != "numpy",
    reason="The regression exercises NumPy overflow handling.",
)


class _IdentityMeasurementModel:
    measurement_matrix = np.array([[1.0]])
    noise_covariance = np.array([[1.0]])


def test_extreme_finite_weights_and_schedule_preserve_ratios():
    max_float = np.finfo(float).max
    particles = np.array([[-1.0], [2.0], [4.0]])
    weights = np.array([max_float, max_float / 2.0, 0.0])
    step_schedule = np.array([max_float, max_float / 2.0])

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        transported, info = gaussian_particle_flow_update(
            particles,
            _IdentityMeasurementModel(),
            np.array([0.5]),
            weights=weights,
            step_schedule=step_schedule,
            jitter=0.0,
            return_info=True,
        )

    assert transported.shape == particles.shape
    assert np.all(np.isfinite(transported))
    npt.assert_allclose(np.asarray(info.mean_trace[0]), np.array([0.0]))
    npt.assert_allclose(info.lambdas, np.array([0.0, 2.0 / 3.0, 1.0]))
