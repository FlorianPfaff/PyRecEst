import unittest

import numpy as np
from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.distributions.cart_prod.cart_prod_stacked_distribution import (
    CartProdStackedDistribution,
)


class TestCartProdStackedTemporalSampleCount(unittest.TestCase):
    def setUp(self):
        self.dist = CartProdStackedDistribution(
            [GaussianDistribution(array([0.0]), eye(1))]
        )

    def test_sample_rejects_numpy_timedelta_counts(self):
        for count in (np.timedelta64(4, "ns"), np.timedelta64(4, "us")):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "n must be an integer"):
                    self.dist.sample(count)


if __name__ == "__main__":
    unittest.main()
