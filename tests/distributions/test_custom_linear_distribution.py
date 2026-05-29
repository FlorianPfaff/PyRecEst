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

    def test_set_mean_shifts_distribution(self):
        g = GaussianDistribution(array([1.0]), eye(1))
        cld = CustomLinearDistribution.from_distribution(g)

        cld.set_mean(array([3.0]))

        npt.assert_allclose(cld.mean(), array([3.0]))
        npt.assert_allclose(cld.shift_by, array([2.0]))
        npt.assert_allclose(cld.pdf(array([3.0])), g.pdf(array([1.0])))

    def test_pdf_accepts_scalar_and_list_inputs(self):
        cld = CustomLinearDistribution(
            lambda xs: xs[:, 0] * 0.0 + 1.0, 1, shift_by=[0.2]
        )

        scalar_pdf = cld.pdf(0.3)
        list_pdf = cld.pdf([0.3, 0.4])
        array_pdf = cld.pdf(array([0.3, 0.4]))

        self.assertEqual(scalar_pdf.shape, (1,))
        npt.assert_allclose(list_pdf, array_pdf)

    def test_pdf_accepts_multidimensional_list_inputs(self):
        cld = CustomLinearDistribution(lambda xs: xs[:, 0], 2, shift_by=[0.2, -0.1])

        list_pdf = cld.pdf([[1.0, 2.0], [3.0, 4.0]])
        array_pdf = cld.pdf(array([[1.0, 2.0], [3.0, 4.0]]))

        npt.assert_allclose(list_pdf, array_pdf)

    def test_constructor_rejects_wrong_shift_shape(self):
        with self.assertRaisesRegex(ValueError, "shift_by"):
            CustomLinearDistribution(lambda xs: xs[:, 0], 2, shift_by=[0.2])

    def test_shift_accepts_list_offset(self):
        cld = CustomLinearDistribution(lambda xs: xs[:, 0], 2)

        shifted = cld.shift([0.2, -0.1])

        npt.assert_allclose(shifted.shift_by, array([0.2, -0.1]))

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
