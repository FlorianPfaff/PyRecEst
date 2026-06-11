import unittest

import numpy as np
from pyrecest.backend import allclose, array, eye
from pyrecest.distributions import GaussianDistribution, VonMisesFisherDistribution
from pyrecest.distributions.cart_prod.abstract_lin_periodic_cart_prod_distribution import (
    AbstractLinPeriodicCartProdDistribution,
)
from pyrecest.distributions.cart_prod.cart_prod_stacked_distribution import (
    CartProdStackedDistribution,
)


class PurePeriodicLinPeriodicDistribution(AbstractLinPeriodicCartProdDistribution):
    def __init__(self):
        super().__init__(bound_dim=1, lin_dim=0)

    @property
    def input_dim(self):
        return self.dim

    def pdf(self, _xs):
        raise NotImplementedError

    def marginalize_linear(self):
        raise NotImplementedError

    def marginalize_periodic(self):
        raise NotImplementedError


class ConcreteCartProdStackedDistribution(CartProdStackedDistribution):
    @property
    def input_dim(self):
        return sum(dist.input_dim for dist in self.dists)

    def get_manifold_size(self):
        return float("inf")


class TestCartProdStackedDistribution(unittest.TestCase):
    def test_lin_periodic_base_rejects_purely_periodic_manifold_size(self):
        dist = PurePeriodicLinPeriodicDistribution()

        with self.assertRaisesRegex(ValueError, "purely periodic"):
            dist.get_manifold_size()

    def test_base_class_is_instantiable_and_reports_geometry(self):
        dist = CartProdStackedDistribution(
            [
                GaussianDistribution(array([0.0, 0.0]), eye(2)),
                GaussianDistribution(array([0.0, 0.0, 0.0]), eye(3)),
            ]
        )

        self.assertEqual(dist.dim, 5)
        self.assertEqual(dist.input_dim, 5)
        self.assertEqual(dist.get_manifold_size(), float("inf"))

    def test_sample_returns_concatenated_component_samples(self):
        dist = ConcreteCartProdStackedDistribution(
            [
                GaussianDistribution(array([0.0, 0.0]), eye(2)),
                GaussianDistribution(array([0.0, 0.0, 0.0]), eye(3)),
            ]
        )

        samples = dist.sample(5)

        self.assertEqual(samples.shape, (5, 5))

    def test_sample_accepts_integer_like_count(self):
        dist = ConcreteCartProdStackedDistribution(
            [
                GaussianDistribution(array([0.0, 0.0]), eye(2)),
                GaussianDistribution(array([0.0, 0.0, 0.0]), eye(3)),
            ]
        )

        samples = dist.sample(np.array(4.0))

        self.assertEqual(samples.shape, (4, 5))

    def test_sample_rejects_invalid_count(self):
        dist = ConcreteCartProdStackedDistribution(
            [
                GaussianDistribution(array([0.0, 0.0]), eye(2)),
                GaussianDistribution(array([0.0, 0.0, 0.0]), eye(3)),
            ]
        )

        for n in (0, -1, 1.5, True, [3]):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    dist.sample(n)

    def test_pdf_slices_component_columns_for_batched_samples(self):
        dist = ConcreteCartProdStackedDistribution(
            [
                GaussianDistribution(array([0.0, 0.0]), eye(2)),
                GaussianDistribution(array([0.0, 0.0, 0.0]), eye(3)),
            ]
        )
        xs = array([[1.0, 2.0, 3.0, 4.0, 5.0], [0.5, 1.5, 2.5, 3.5, 4.5]])

        pdf_values = dist.pdf(xs)
        expected = dist.dists[0].pdf(xs[:, :2]) * dist.dists[1].pdf(xs[:, 2:])

        self.assertEqual(pdf_values.shape, (2,))
        self.assertTrue(allclose(pdf_values, expected))

    def test_pdf_slices_component_entries_for_single_sample(self):
        dist = ConcreteCartProdStackedDistribution(
            [
                GaussianDistribution(array([0.0, 0.0]), eye(2)),
                GaussianDistribution(array([0.0, 0.0, 0.0]), eye(3)),
            ]
        )
        x = array([1.0, 2.0, 3.0, 4.0, 5.0])

        pdf_value = dist.pdf(x)
        expected = dist.dists[0].pdf(x[:2]) * dist.dists[1].pdf(x[2:])

        self.assertTrue(allclose(pdf_value, expected))

    def test_pdf_accepts_list_inputs(self):
        dist = ConcreteCartProdStackedDistribution(
            [
                GaussianDistribution(array([0.0, 0.0]), eye(2)),
                GaussianDistribution(array([0.0]), eye(1)),
            ]
        )
        single = [1.0, 2.0, 3.0]
        batched = [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]

        self.assertTrue(allclose(dist.pdf(single), dist.pdf(array(single))))
        self.assertTrue(allclose(dist.pdf(batched), dist.pdf(array(batched))))

    def test_set_mode_uses_component_input_dimensions(self):
        dist = CartProdStackedDistribution(
            [
                GaussianDistribution(array([1.0]), eye(1)),
                VonMisesFisherDistribution(array([0.0, 0.0, 1.0]), 2.0),
            ]
        )

        shifted = dist.set_mode(array([2.0, 1.0, 0.0, 0.0]))

        self.assertTrue(allclose(shifted.dists[0].mu, array([2.0])))
        self.assertTrue(allclose(shifted.dists[1].mu, array([1.0, 0.0, 0.0])))

    def test_shift_uses_cumulative_component_dimensions(self):
        dist = ConcreteCartProdStackedDistribution(
            [
                GaussianDistribution(array([1.0, 2.0]), eye(2)),
                GaussianDistribution(array([3.0, 4.0, 5.0]), eye(3)),
            ]
        )

        shifted = dist.shift(array([10.0, 20.0, 30.0, 40.0, 50.0]))

        self.assertIsInstance(shifted, ConcreteCartProdStackedDistribution)
        self.assertTrue(allclose(shifted.dists[0].mu, array([11.0, 22.0])))
        self.assertTrue(allclose(shifted.dists[1].mu, array([33.0, 44.0, 55.0])))

    def test_shift_rejects_wrong_offset_count(self):
        dist = ConcreteCartProdStackedDistribution(
            [
                GaussianDistribution(array([1.0, 2.0]), eye(2)),
                GaussianDistribution(array([3.0, 4.0, 5.0]), eye(3)),
            ]
        )

        with self.assertRaisesRegex(ValueError, "offsets"):
            dist.shift(array([1.0, 2.0]))

    def test_set_mode_preserves_concrete_distribution_type(self):
        dist = ConcreteCartProdStackedDistribution(
            [
                GaussianDistribution(array([1.0, 2.0]), eye(2)),
                GaussianDistribution(array([3.0, 4.0, 5.0]), eye(3)),
            ]
        )

        shifted = dist.set_mode(array([10.0, 20.0, 30.0, 40.0, 50.0]))

        self.assertIsInstance(shifted, ConcreteCartProdStackedDistribution)
        self.assertTrue(allclose(shifted.dists[0].mu, array([10.0, 20.0])))
        self.assertTrue(allclose(shifted.dists[1].mu, array([30.0, 40.0, 50.0])))


if __name__ == "__main__":
    unittest.main()
