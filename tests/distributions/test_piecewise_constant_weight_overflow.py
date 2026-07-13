import numpy as np
import numpy.testing as npt

from pyrecest.backend import array, to_numpy
from pyrecest.distributions.circle.piecewise_constant_distribution import (
    PiecewiseConstantDistribution,
)


def test_maximum_finite_weights_normalize_without_overflow():
    backend_dtype = np.asarray(to_numpy(array([1.0], dtype=float))).dtype
    maximum_finite_weight = np.finfo(backend_dtype).max

    distribution = PiecewiseConstantDistribution(
        array([maximum_finite_weight, maximum_finite_weight / 2.0], dtype=float)
    )

    npt.assert_allclose(
        np.asarray(to_numpy(distribution.w)),
        np.array([2.0 / (3.0 * np.pi), 1.0 / (3.0 * np.pi)]),
        rtol=1.0e-6,
        atol=0.0,
    )
