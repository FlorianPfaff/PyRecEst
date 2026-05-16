import unittest
from warnings import catch_warnings, simplefilter

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, column_stack, diag, linspace, meshgrid, reshape
from pyrecest.distributions import GaussianDistribution
from pyrecest.distributions.nonperiodic.gaussian_mixture import GaussianMixture
from pyrecest.distributions.nonperiodic.linear_mixture import LinearMixture


class LinearMixtureTest(unittest.TestCase):
    def test_constructor_warning(self):
        with catch_warnings(record=True) as w:
            simplefilter("always")
            LinearMixture(
                [
                    GaussianDistribution(array([1.0]), array([[1.0]])),
                    GaussianDistribution(array([50.0]), array([[1.0]])),
                ],
                array([0.3, 0.7]),
            )
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertIn(
                "For mixtures of Gaussians, consider using GaussianMixture.",
                str(w[-1].message),
            )

    def test_prunes_zero_weight_components(self):
        gm1 = GaussianDistribution(array([1.0]), array([[1.0]]))
        gm2 = GaussianDistribution(array([50.0]), array([[1.0]]))
        gm3 = GaussianDistribution(array([100.0]), array([[1.0]]))

        with catch_warnings():
            simplefilter("ignore", category=UserWarning)
            lm = LinearMixture([gm1, gm2, gm3], array([0.3, 0.0, 0.7]))

        self.assertEqual(len(lm.dists), 2)
        npt.assert_allclose(lm.w, array([0.3, 0.7]))
        npt.assert_allclose(lm.dists[0].mu, gm1.mu)
        npt.assert_allclose(lm.dists[1].mu, gm3.mu)

    def test_rejects_all_zero_weights(self):
        gm1 = GaussianDistribution(array([1.0]), array([[1.0]]))
        gm2 = GaussianDistribution(array([50.0]), array([[1.0]]))

        with catch_warnings():
            simplefilter("ignore", category=UserWarning)
            with self.assertRaises(ValueError):
                LinearMixture([gm1, gm2], array([0.0, 0.0]))

    def test_pdf(self):
        gm1 = GaussianDistribution(array([1.0, 1.0]), diag(array([2.0, 3.0])))
        gm2 = GaussianDistribution(-array([3.0, 1.0]), diag(array([2.0, 3.0])))

        with catch_warnings():
            simplefilter("ignore", category=UserWarning)
            lm = LinearMixture([gm1, gm2], array([0.3, 0.7]))

        x, y = meshgrid(linspace(-2, 2, 100), linspace(-2, 2, 100), indexing="ij")
        points = column_stack((x.ravel(), y.ravel()))

        npt.assert_allclose(
            lm.pdf(points), 0.3 * gm1.pdf(points) + 0.7 * gm2.pdf(points), atol=1e-20
        )

    def test_pdf_accepts_vectorized_one_dimensional_inputs(self):
        gm1 = GaussianDistribution(array([1.0]), array([[2.0]]))
        gm2 = GaussianDistribution(-array([3.0]), array([[3.0]]))

        with catch_warnings():
            simplefilter("ignore", category=UserWarning)
            lm = LinearMixture([gm1, gm2], array([0.3, 0.7]))

        xs = linspace(-2.0, 2.0, 100)
        expected = 0.3 * gm1.pdf(xs) + 0.7 * gm2.pdf(xs)
        npt.assert_allclose(lm.pdf(xs), expected, atol=1e-20)
        npt.assert_allclose(lm.pdf(reshape(xs, (-1, 1))), expected, atol=1e-20)

    def test_gaussian_mixture_pdf_accepts_vectorized_one_dimensional_inputs(self):
        gm1 = GaussianDistribution(array([1.0]), array([[2.0]]))
        gm2 = GaussianDistribution(-array([3.0]), array([[3.0]]))
        gmix = GaussianMixture([gm1, gm2], array([0.3, 0.7]))

        xs = linspace(-2.0, 2.0, 100)
        expected = 0.3 * gm1.pdf(xs) + 0.7 * gm2.pdf(xs)
        npt.assert_allclose(gmix.pdf(xs), expected, atol=1e-20)
        npt.assert_allclose(gmix.pdf(reshape(xs, (-1, 1))), expected, atol=1e-20)


if __name__ == "__main__":
    unittest.main()
