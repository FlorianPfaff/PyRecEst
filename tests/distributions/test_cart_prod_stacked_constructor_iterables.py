import unittest

from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.distributions.cart_prod.cart_prod_stacked_distribution import (
    CartProdStackedDistribution,
)


class TestCartProdStackedConstructorIterables(unittest.TestCase):
    @staticmethod
    def _components():
        return [
            GaussianDistribution(array([1.0]), eye(1)),
            GaussianDistribution(array([2.0, 3.0]), eye(2)),
        ]

    def test_constructor_materializes_one_shot_iterable(self):
        components = self._components()

        dist = CartProdStackedDistribution(component for component in components)

        self.assertEqual(dist.dim, 3)
        self.assertEqual(dist.input_dim, 3)
        self.assertEqual(len(dist.dists), 2)
        self.assertEqual(dist.mean().shape, (3,))

    def test_constructor_does_not_alias_mutable_component_container(self):
        components = self._components()[:1]
        dist = CartProdStackedDistribution(components)

        components.append(self._components()[1])

        self.assertEqual(dist.dim, 1)
        self.assertEqual(dist.input_dim, 1)
        self.assertEqual(len(dist.dists), 1)


if __name__ == "__main__":
    unittest.main()
