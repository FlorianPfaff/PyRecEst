import numpy.testing as npt
import pyrecest.backend
import pytest

from pyrecest.backend import array, eye, to_numpy
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.hyperspherical_ukf import HypersphericalUKF


pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ == "jax",
    reason="HypersphericalUKF is unsupported on JAX.",
)


def test_filter_state_assignment_does_not_alias_caller_state():
    filter_ = HypersphericalUKF(dim=2)
    assigned = GaussianDistribution(array([1.0, 0.0]), eye(2))

    filter_.filter_state = assigned
    expected_mean = to_numpy(filter_.filter_state.mu).copy()
    expected_covariance = to_numpy(filter_.filter_state.C).copy()
    assert filter_.filter_state is not assigned

    assigned.mu = array([0.0, 1.0])
    assigned.C = 2.0 * eye(2)

    npt.assert_allclose(to_numpy(filter_.filter_state.mu), expected_mean)
    npt.assert_allclose(to_numpy(filter_.filter_state.C), expected_covariance)
