import warnings

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, to_numpy
from pyrecest.distributions import LinearDiracDistribution


def _active_max_finite():
    active_dtype = to_numpy(array([0.0], dtype=float)).dtype
    return np.finfo(active_dtype).max


def test_normalizes_extreme_finite_weights_without_overflow():
    max_finite = _active_max_finite()

    with warnings.catch_warnings(), np.errstate(
        over="raise", invalid="raise", divide="raise"
    ):
        warnings.simplefilter("ignore", RuntimeWarning)
        dist = LinearDiracDistribution(
            array([[0.0], [1.0]]),
            array([max_finite, max_finite / 2.0], dtype=float),
        )

    npt.assert_allclose(to_numpy(dist.w), [2.0 / 3.0, 1.0 / 3.0])


def test_weighted_moments_accept_extreme_finite_weights():
    max_finite = _active_max_finite()

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        mean, covariance = LinearDiracDistribution.weighted_samples_to_mean_and_cov(
            array([0.0, 3.0]),
            array([max_finite, max_finite / 2.0], dtype=float),
        )

    npt.assert_allclose(to_numpy(mean), [1.0])
    npt.assert_allclose(to_numpy(covariance), [[2.0]])
