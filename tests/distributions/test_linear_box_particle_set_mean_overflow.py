"""Regression tests for overflow-safe box-particle mean shifts."""

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, to_numpy
from pyrecest.distributions.nonperiodic.linear_box_particle_distribution import (
    LinearBoxParticleDistribution,
)


def test_set_mean_avoids_overflow_when_finite_shifted_support_exists():
    backend_dtype = to_numpy(array([1.0])).dtype
    max_finite = np.finfo(backend_dtype).max
    lower_value = 0.75 * max_finite
    upper_value = max_finite
    target_mean = -0.5 * max_finite
    dist = LinearBoxParticleDistribution(array([[lower_value]]), array([[upper_value]]))

    with np.errstate(over="raise", invalid="raise"):
        shifted = dist.set_mean(array([target_mean]))

    shifted_lower = to_numpy(shifted.lower)
    shifted_upper = to_numpy(shifted.upper)
    shifted_mean = to_numpy(shifted.mean())

    assert np.isfinite(shifted_lower).all()
    assert np.isfinite(shifted_upper).all()
    npt.assert_allclose(
        shifted_lower,
        np.array([[-0.625 * max_finite]], dtype=backend_dtype),
        rtol=5e-7,
    )
    npt.assert_allclose(
        shifted_upper,
        np.array([[-0.375 * max_finite]], dtype=backend_dtype),
        rtol=5e-7,
    )
    npt.assert_allclose(
        shifted_mean,
        np.array([target_mean], dtype=backend_dtype),
        rtol=5e-7,
    )
