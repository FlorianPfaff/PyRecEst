import unittest

from pyrecest.backend import (
    allclose,
    array,
    diag,
    diff,
    hstack,
    linalg,
    ones,
    random,
    zeros,
)
from pyrecest.distributions import (
    GaussianDistribution,
    HyperhemisphericalUniformDistribution,
    HyperhemisphericalWatsonDistribution,
)
from pyrecest.distributions.cart_prod.se3_lin_vel_cart_prod_stacked_distribution import (
    SE3LinVelCartProdStackedDistribution,
)


class TestSE3LinVelCartProdStackedDistribution(unittest.TestCase):
    def test_constructor(self):
        SE3LinVelCartProdStackedDistribution(
            [
                HyperhemisphericalUniformDistribution(3),
                GaussianDistribution(
                    array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                    diag(array([3.0, 2.0, 1.0, 4.0, 3.0, 4.0])),
                ),
            ]
        )

    def test_sampling(self):
        cpd = SE3LinVelCartProdStackedDistribution(
            [
                HyperhemisphericalUniformDistribution(3),
                GaussianDistribution(
                    array([1.0, 2.0, 0.0, -2.0, -1.0, 3.0]),
                    diag(array([3.0, 2.0, 3.0, 3.0, 4.0, 5.0])),
                ),
            ]
        )
        samples = cpd.sample(100)
        self.assertEqual(samples.shape, (100, 10))

    def test_pdf(self):
        cpd = SE3LinVelCartProdStackedDistribution(
            [
                HyperhemisphericalUniformDistribution(3),
                GaussianDistribution(
                    array([1.0, 2.0, 0.0, -2.0, -1.0, 3.0]),
                    diag(array([3.0, 2.0, 3.0, 3.0, 4.0, 5.0])),
                ),
            ]
        )
        self.assertEqual(cpd.pdf(random.normal(size=(100, 10))).shape, (100,))

        pdf_values = cpd.pdf(ones((100, 10)))
        self.assertTrue(allclose(diff(pdf_values), zeros(99)))

    def test_mode(self):
        mu = array([2.0, 1.0, 3.0, 1.0])
        watson = HyperhemisphericalWatsonDistribution(mu / linalg.norm(mu), 2)
        gaussian = GaussianDistribution(
            array([1.0, 2.0, 0.0, -2.0, -1.0, 3.0]),
            diag(array([3.0, 2.0, 3.0, 3.0, 4.0, 5.0])),
        )
        cpd = SE3LinVelCartProdStackedDistribution([watson, gaussian])
        self.assertTrue(allclose(cpd.mode(), hstack([watson.mode(), gaussian.mode()])))


if __name__ == "__main__":
    unittest.main()
