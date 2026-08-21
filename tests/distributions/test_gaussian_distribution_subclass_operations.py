"""Regression tests for Gaussian subclass preservation."""

import unittest

from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution


class _DerivedGaussian(GaussianDistribution):
    """Minimal derived Gaussian used to verify operation return types."""


class TestGaussianDistributionSubclassOperations(unittest.TestCase):
    def setUp(self):
        self.derived = _DerivedGaussian(array([0.0]), array([[2.0]]))
        self.other = GaussianDistribution(array([1.0]), array([[3.0]]))

    def test_multiply_preserves_left_hand_subclass(self):
        result = self.derived.multiply(self.other)

        self.assertIsInstance(result, _DerivedGaussian)

    def test_convolve_preserves_left_hand_subclass(self):
        result = self.derived.convolve(self.other)

        self.assertIsInstance(result, _DerivedGaussian)


if __name__ == "__main__":
    unittest.main()
