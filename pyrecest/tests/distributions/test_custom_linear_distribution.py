import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, concatenate, eye, linspace, meshgrid, pi
from pyrecest.distributions import CustomLinearDistribution, GaussianDistribution
from pyrecest.distributions.nonperiodic.gaussian_mixture import GaussianMixture


class CustomLinearDistributionTest(unittest.TestCase):
    def setUp(self):
        g1 = GaussianDistribution(array([1.0, 1.0]), eye(2))
        g2 = GaussianDistribution(array([-1.0, -1.0]), eye(2))
        self.gm = GaussianMixture([g1, g2], array([0.7, 0.3]))

    def test_init_and_mean(self):
        cld = CustomLinearDistribution.from_distribution(self.gm)
        self.verify_pdf_equal(cld, self.gm, 1e-14)

    def test_integrate(self):
        cld = CustomLinearDistribution.from_distribution(self.gm)
        self.assertAlmostEqual(cld.integrate(), 1.0, delta=1e-7)

    def test_normalize(self):
        self.gm.w = self.gm.w / 2
        cld = CustomLinearDistribution.from_distribution(self.gm)
        self.assertAlmostEqual(cld.integrate(), 0.5, delta=1e-8)

    @staticmethod
    def verify_pdf_equal(dist1, dist2, tol):
        x, y = meshgrid(
            linspace(0.0, 2.0 * pi, 10), linspace(0.0, 2.0 * pi, 10), indexing="ij"
        )
        npt.assert_allclose(
            dist1.pdf(concatenate((x, y)).reshape(2, -1).T),
            dist2.pdf(concatenate((x, y)).reshape(2, -1).T),
            atol=tol,
        )


if __name__ == "__main__":
    unittest.main()
