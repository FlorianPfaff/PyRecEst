import numpy as np
import numpy.testing as npt

from pyrecest.backend import array, to_numpy
from pyrecest.distributions import PiecewiseConstantDistribution


def test_max_finite_weights_normalize_without_overflow():
    probe = np.asarray(to_numpy(array([0.0], dtype=float)))
    maximum = np.finfo(probe.dtype).max

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        distribution = PiecewiseConstantDistribution(
            array([maximum, maximum / 2.0], dtype=float)
        )

    actual = np.asarray(to_numpy(distribution.w), dtype=float)
    expected = np.array([2.0, 1.0]) / (3.0 * np.pi)

    npt.assert_allclose(actual, expected, rtol=1e-6, atol=0.0)
    npt.assert_allclose(np.mean(actual) * 2.0 * np.pi, 1.0, rtol=1e-6)
