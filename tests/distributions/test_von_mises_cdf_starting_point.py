import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array
from pyrecest.distributions import VonMisesDistribution


@unittest.skipIf(
    pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
    reason="Von Mises CDF is not supported on this backend",
)
class TestVonMisesCdfStartingPoint(unittest.TestCase):
    def test_rejects_non_scalar_starting_point(self):
        dist = VonMisesDistribution(0.3, 1.2)

        for starting_point in ([0.0, 1.0], array([0.0, 1.0])):
            with self.subTest(starting_point=starting_point):
                with self.assertRaisesRegex(
                    ValueError, "starting_point must be a scalar"
                ):
                    dist.cdf(array([0.5, 1.0]), starting_point=starting_point)

    def test_rejects_non_finite_starting_point(self):
        dist = VonMisesDistribution(0.3, 1.2)

        for starting_point in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(starting_point=starting_point):
                with self.assertRaisesRegex(
                    ValueError, "starting_point must be finite"
                ):
                    dist.cdf(array([0.5, 1.0]), starting_point=starting_point)

    def test_accepts_singleton_and_numpy_scalar_starting_points(self):
        dist = VonMisesDistribution(0.3, 1.2)
        xs = array([0.5, 1.0])

        expected = dist.cdf(xs, starting_point=0.25)
        npt.assert_allclose(dist.cdf(xs, starting_point=array([0.25])), expected)
        npt.assert_allclose(dist.cdf(xs, starting_point=np.float64(0.25)), expected)


if __name__ == "__main__":
    unittest.main()
