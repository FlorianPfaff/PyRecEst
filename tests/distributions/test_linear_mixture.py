import unittest
from warnings import catch_warnings, simplefilter

import numpy as np
import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    array,
    column_stack,
    diag,
    linspace,
    meshgrid,
    reshape,
    zeros,
)
from pyrecest.distributions import GaussianDistribution
from pyrecest.distributions.nonperiodic.abstract_linear_distribution import (
    AbstractLinearDistribution,
)
from pyrecest.distributions.nonperiodic.gaussian_mixture import GaussianMixture
from pyrecest.distributions.nonperiodic.linear_dirac_distribution import (
    LinearDiracDistribution,
)
from pyrecest.distributions.nonperiodic.linear_mixture import LinearMixture


class _ConstantLinearDistribution(AbstractLinearDistribution):
    def __init__(self, value):
        super().__init__(1)
        self.value = value

    def pdf(self, xs):
        xs = array(xs)
        if xs.ndim == 0:
            return array(0.0)
        if xs.ndim == 1:
            return zeros(xs.shape)
        return zeros(xs.shape[0])

    def sample(self, n):
        return array([[self.value]] * int(n))


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

    def test_constructor_rejects_invalid_distribution_list(self):
        invalid_cases = (
            ([], array([]), "at least one"),
            ([object()], array([1.0]), "linear distributions"),
        )

        for dists, weights, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    LinearMixture(dists, weights)

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

    def test_pdf_rejects_wrong_point_dimension(self):
        gm1 = GaussianDistribution(array([0.0, 0.0]), diag(array([1.0, 1.0])))
        gm2 = GaussianDistribution(array([1.0, 1.0]), diag(array([1.0, 1.0])))
        gmix = GaussianMixture([gm1, gm2], array([0.25, 0.75]))

        with self.assertRaisesRegex(ValueError, "Dimension mismatch"):
            gmix.pdf(array([1.0, 2.0, 3.0, 4.0]))

    def test_gaussian_mixture_pdf_accepts_vectorized_one_dimensional_inputs(self):
        gm1 = GaussianDistribution(array([1.0]), array([[2.0]]))
        gm2 = GaussianDistribution(-array([3.0]), array([[3.0]]))
        gmix = GaussianMixture([gm1, gm2], array([0.3, 0.7]))

        xs = linspace(-2.0, 2.0, 100)
        expected = 0.3 * gm1.pdf(xs) + 0.7 * gm2.pdf(xs)
        npt.assert_allclose(gmix.pdf(xs), expected, atol=1e-20)
        npt.assert_allclose(gmix.pdf(reshape(xs, (-1, 1))), expected, atol=1e-20)

    def test_sample_accepts_flat_one_dimensional_dirac_components(self):
        dirac = LinearDiracDistribution(array([1.0, 2.0, 3.0]), array([0.2, 0.5, 0.3]))
        lm = LinearMixture([dirac], array([1.0]))

        samples = lm.sample(5)

        self.assertEqual(samples.shape, (5, 1))

    def test_sample_accepts_integer_like_count(self):
        mixture = GaussianMixture(
            [
                GaussianDistribution(array([0.0, 0.0]), diag(array([1.0, 1.0]))),
                GaussianDistribution(array([1.0, 1.0]), diag(array([1.0, 1.0]))),
            ],
            array([0.25, 0.75]),
        )

        samples = mixture.sample(np.array(4.0))

        self.assertEqual(samples.shape, (4, 2))

    def test_sample_preserves_numpy_drawn_component_order(self):
        if pyrecest.backend.__backend_name__ != "numpy":
            self.skipTest("NumPy RNG regression test")

        mixture = LinearMixture(
            [_ConstantLinearDistribution(0.1), _ConstantLinearDistribution(0.9)],
            array([0.5, 0.5]),
        )

        pyrecest.backend.random.seed(0)
        samples = mixture.sample(8)

        self.assertEqual(samples.shape, (8, 1))
        npt.assert_allclose(
            samples,
            array([[0.9], [0.9], [0.9], [0.9], [0.1], [0.9], [0.1], [0.9]]),
        )

    def test_sample_rejects_invalid_count(self):
        mixture = GaussianMixture(
            [
                GaussianDistribution(array([0.0, 0.0]), diag(array([1.0, 1.0]))),
                GaussianDistribution(array([1.0, 1.0]), diag(array([1.0, 1.0]))),
            ],
            array([0.25, 0.75]),
        )

        for n in (0, -1, 1.5, True, [3]):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    mixture.sample(n)


if __name__ == "__main__":
    unittest.main()
