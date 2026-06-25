"""Regression tests for CircularUKF boolean flag validation."""

import pyrecest.backend
import pytest
from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.circular_ukf import CircularUKF


def _identity(angle):
    return angle


def test_update_nonlinear_rejects_non_boolean_measurement_periodic():
    if pyrecest.backend.__backend_name__ in ("pytorch", "jax"):
        pytest.skip("Not supported on this backend")

    filt = CircularUKF()
    prior = GaussianDistribution(array([0.5]), array([[0.7]]))
    meas_noise = GaussianDistribution(array([0.0]), array([[0.7]]))

    filt.filter_state = prior
    with pytest.raises(TypeError, match="measurement_periodic"):
        filt.update_nonlinear(
            _identity,
            meas_noise,
            prior.mu,
            measurement_periodic="False",
        )
