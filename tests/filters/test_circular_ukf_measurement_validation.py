"""Regression tests for CircularUKF measurement numeric validation."""

import numpy as np
import pyrecest.backend
import pytest
from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.circular_ukf import CircularUKF


def _measurement(angle):
    return angle


@pytest.fixture
def circular_filter():
    filt = CircularUKF()
    filt.filter_state = GaussianDistribution(array([0.5]), array([[0.7]]))
    return filt


@pytest.fixture
def measurement_noise():
    return GaussianDistribution(array([0.0]), array([[0.7]]))


@pytest.mark.parametrize("z", [True, "1.0", b"1.0", 1.0 + 0.0j])
def test_update_identity_rejects_ambiguous_scalar_measurements(
    circular_filter, measurement_noise, z
):
    with pytest.raises(TypeError, match="measurement z.*real numeric"):
        circular_filter.update_identity(measurement_noise, z)


@pytest.mark.skipif(
    pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
    reason="Not supported on this backend",
)
@pytest.mark.parametrize(
    "z",
    [
        [True],
        ["1.0"],
        [b"1.0"],
        [1.0 + 0.0j],
        np.asarray(["1.0"], dtype=object),
    ],
)
def test_update_nonlinear_rejects_ambiguous_measurement_vectors(
    circular_filter, measurement_noise, z
):
    with pytest.raises(TypeError, match="measurement vector.*real numeric"):
        circular_filter.update_nonlinear(_measurement, measurement_noise, z)


@pytest.mark.skipif(
    pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
    reason="Not supported on this backend",
)
def test_update_nonlinear_rejects_ambiguous_measurement_function_outputs(
    circular_filter, measurement_noise
):
    with pytest.raises(TypeError, match="measurement vector.*real numeric"):
        circular_filter.update_nonlinear(
            lambda _angle: ["1.0"], measurement_noise, [0.5]
        )
