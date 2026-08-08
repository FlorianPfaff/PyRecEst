import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array, float64
from pyrecest.distributions import ToroidalDiracDistribution


def _as_numpy(value):
    return np.asarray(pyrecest.backend.to_numpy(value))


class TestToroidalDiracCorrelationStability(unittest.TestCase):
    def test_tiny_correlated_spread_does_not_underflow(self):
        tiny = 1e-200
        dist = ToroidalDiracDistribution(
            array([[0.0, 0.0], [tiny, tiny]], dtype=float64),
            array([0.5, 0.5], dtype=float64),
        )

        correlation = _as_numpy(dist.circular_correlation_jammalamadaka())

        self.assertTrue(np.isfinite(correlation))
        npt.assert_allclose(correlation, 1.0, rtol=1e-12, atol=1e-12)

    def test_zero_circular_variance_is_reported_explicitly(self):
        dist = ToroidalDiracDistribution(
            array([[0.0, 0.0], [0.0, 1.0]], dtype=float64),
            array([0.5, 0.5], dtype=float64),
        )

        with self.assertRaisesRegex(ValueError, "zero circular variance"):
            dist.circular_correlation_jammalamadaka()


if __name__ == "__main__":
    unittest.main()
