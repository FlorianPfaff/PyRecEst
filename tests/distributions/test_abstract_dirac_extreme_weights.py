import warnings

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, to_numpy
from pyrecest.distributions import LinearDiracDistribution


def test_normalizes_extreme_finite_weights_without_overflow():
    active_dtype = to_numpy(array([0.0], dtype=float)).dtype
    max_finite = np.finfo(active_dtype).max

    with warnings.catch_warnings(), np.errstate(
        over="raise", invalid="raise", divide="raise"
    ):
        warnings.simplefilter("ignore", RuntimeWarning)
        dist = LinearDiracDistribution(
            array([[0.0], [1.0]]),
            array([max_finite, max_finite / 2.0], dtype=float),
        )

    npt.assert_allclose(to_numpy(dist.w), [2.0 / 3.0, 1.0 / 3.0])
