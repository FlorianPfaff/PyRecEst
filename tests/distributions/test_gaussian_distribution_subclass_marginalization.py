"""Regression test for Gaussian subclass preservation during marginalization."""

import unittest

from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution


class _DerivedGaussian(GaussianDistribution):
    """Minimal derived Gaussian used to verify operation return types."""


class TestGaussianDistributionSubclassMarginalization(unittest.TestCase):
    def test_marginalize_out_preserves_subclass(self):
        derived = _DerivedGaussian(
            array([1.0, 2.0]),
            array([[2.0, 0.5], [0.5, 3.0]]),
        )

        result = derived.marginalize_out(1)

        self.assertIsInstance(result, _DerivedGaussian)
        self.assertEqual(result.dim, 1)


if __name__ == "__main__":
    unittest.main()
