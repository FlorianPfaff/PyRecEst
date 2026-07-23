import unittest

from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution, VonMisesDistribution
from pyrecest.distributions.cart_prod.cart_prod_stacked_distribution import (
    CartProdStackedDistribution,
)


class TestCartProdStackedScalarSampling(unittest.TestCase):
    def test_sample_keeps_scalar_components_in_separate_columns(self):
        dist = CartProdStackedDistribution(
            [
                VonMisesDistribution(0.0, 1.0),
                VonMisesDistribution(1.0, 2.0),
            ]
        )

        samples = dist.sample(5)

        self.assertEqual(samples.shape, (5, 2))

    def test_sample_combines_scalar_and_vector_components(self):
        dist = CartProdStackedDistribution(
            [
                VonMisesDistribution(0.0, 1.0),
                GaussianDistribution(array([0.0, 0.0]), eye(2)),
            ]
        )

        samples = dist.sample(5)

        self.assertEqual(samples.shape, (5, 3))


if __name__ == "__main__":
    unittest.main()
