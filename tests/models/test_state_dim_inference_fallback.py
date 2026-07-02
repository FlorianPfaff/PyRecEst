import unittest

from pyrecest.backend import array
from pyrecest.models.validation import infer_state_dim_from_distribution


class TestStateDimInferenceFallback(unittest.TestCase):
    def test_infer_state_dim_skips_uncoercible_dim_attribute(self):
        class UncoercibleDimension:
            def __array__(self, dtype=None, copy=None):
                raise TypeError("not coercible as a scalar dimension")

        class DistributionWithFallbackMean:
            dim = UncoercibleDimension()
            mu = array([0.0, 1.0, 2.0])

        self.assertEqual(infer_state_dim_from_distribution(DistributionWithFallbackMean()), 3)


if __name__ == "__main__":
    unittest.main()
