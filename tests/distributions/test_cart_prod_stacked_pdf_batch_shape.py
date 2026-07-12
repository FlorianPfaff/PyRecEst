import unittest

from pyrecest.backend import allclose, array, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.distributions.cart_prod.cart_prod_stacked_distribution import (
    CartProdStackedDistribution,
)


class TestCartProdStackedPdfBatchShape(unittest.TestCase):
    def setUp(self):
        self.dist = CartProdStackedDistribution(
            [
                GaussianDistribution(array([0.0, 0.0]), eye(2)),
                GaussianDistribution(array([0.0, 0.0, 0.0]), eye(3)),
            ]
        )

    def test_pdf_preserves_arbitrary_leading_batch_dimensions(self):
        xs = array(
            [
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [0.5, 1.5, 2.5, 3.5, 4.5],
                ],
                [
                    [-1.0, 0.0, 1.0, 2.0, 3.0],
                    [2.0, 1.0, 0.0, -1.0, -2.0],
                ],
            ]
        )

        actual = self.dist.pdf(xs)
        expected = self.dist.dists[0].pdf(xs[..., :2]) * self.dist.dists[1].pdf(
            xs[..., 2:]
        )

        self.assertEqual(actual.shape, (2, 2))
        self.assertTrue(allclose(actual, expected))

    def test_pdf_rejects_surplus_event_coordinates(self):
        invalid_points = (
            array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
        )

        for xs in invalid_points:
            with self.subTest(shape=xs.shape):
                with self.assertRaisesRegex(ValueError, "trailing dimension 5"):
                    self.dist.pdf(xs)


if __name__ == "__main__":
    unittest.main()
