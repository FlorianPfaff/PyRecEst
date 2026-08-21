import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, pi, sum, to_numpy
from pyrecest.distributions.circle.piecewise_constant_distribution import (
    PiecewiseConstantDistribution,
)


def _extreme_finite_weights():
    backend_dtype = to_numpy(array([1.0])).dtype
    largest = np.finfo(backend_dtype).max
    return array([largest, largest / 2.0, 0.0])


def test_constructor_normalizes_extreme_finite_weights_without_overflow():
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        dist = PiecewiseConstantDistribution(_extreme_finite_weights())

    npt.assert_allclose(
        to_numpy(dist.w),
        np.array([1.0 / np.pi, 1.0 / (2.0 * np.pi), 0.0]),
        rtol=1e-6,
        atol=0.0,
    )
    npt.assert_allclose(
        float(to_numpy(sum(dist.w) * (2.0 * pi / 3.0))),
        1.0,
        rtol=1e-6,
        atol=0.0,
    )
