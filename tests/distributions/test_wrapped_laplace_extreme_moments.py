import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.distributions.circle.wrapped_laplace_distribution import (
    WrappedLaplaceDistribution,
)


@unittest.skipUnless(
    pyrecest.backend.__backend_name__ == "numpy",
    reason="Strict NumPy floating-point regression",
)
class WrappedLaplaceExtremeMomentTest(unittest.TestCase):
    def test_preserves_representable_moment_with_subnormal_component_rate(self):
        lambda_ = 1.0e-154
        kappa = 1.0e-155
        positive_rate = lambda_ * kappa
        negative_rate = lambda_ / kappa
        distribution = WrappedLaplaceDistribution(lambda_, kappa)

        expected = (
            positive_rate / (positive_rate - 1j) * negative_rate / (negative_rate + 1j)
        )
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            actual = distribution.trigonometric_moment(1)

        self.assertTrue(np.isfinite(actual))
        self.assertNotEqual(actual, 0.0j)
        npt.assert_allclose(actual, expected, rtol=1.0e-12, atol=0.0)


if __name__ == "__main__":
    unittest.main()
