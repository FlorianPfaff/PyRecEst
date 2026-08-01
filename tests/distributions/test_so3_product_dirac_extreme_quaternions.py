import math
import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, float64, linalg, ones
from pyrecest.distributions import SO3ProductDiracDistribution


class SO3ProductDiracExtremeQuaternionTest(unittest.TestCase):
    def test_constructor_normalizes_extreme_finite_quaternion(self):
        component = 1.0e308
        locations = array(
            [[[component, component, 0.0, 0.0]]],
            dtype=float64,
        )
        expected = array(
            [[[math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]]],
            dtype=float64,
        )

        with np.errstate(over="raise", invalid="raise"):
            distribution = SO3ProductDiracDistribution(locations)

        npt.assert_allclose(distribution.d, expected, rtol=1.0e-15, atol=0.0)
        npt.assert_allclose(
            linalg.norm(distribution.d, axis=-1),
            ones((1, 1), dtype=float64),
            rtol=1.0e-15,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
