"""Test for uniform distribution on the hypersphere"""

# pylint: disable=no-name-in-module,no-member
import math
import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array, linalg, log, ones, random
from pyrecest.distributions import (
    AbstractHypersphericalDistribution,
    HypersphericalUniformDistribution,
)


class HypersphericalUniformDistributionTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    def test_integrate_2d(self):
        hud = HypersphericalUniformDistribution(2)
        npt.assert_allclose(hud.integrate(), 1, atol=1e-6)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    def test_integrate_3d(self):
        hud = HypersphericalUniformDistribution(3)
        npt.assert_allclose(hud.integrate(), 1, atol=1e-6)

    def test_pdf(self):
        random.seed(0)
        for dim in range(2, 5):
            hud = HypersphericalUniformDistribution(dim)
            x = random.uniform(size=(dim + 1,))
            x = x / linalg.norm(x)
            npt.assert_allclose(
                hud.pdf(x),
                1
                / AbstractHypersphericalDistribution.compute_unit_hypersphere_surface(
                    dim
                ),
                atol=1e-10,
            )

    def test_pdf_accepts_list_inputs(self):
        hud = HypersphericalUniformDistribution(2)
        points = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

        npt.assert_allclose(hud.pdf(points), hud.pdf(array(points)))
        npt.assert_allclose(hud.pdf(points[0]), hud.pdf(array(points[0])))

    def test_pdf_rejects_wrong_dimension(self):
        hud = HypersphericalUniformDistribution(2)

        with self.assertRaises(ValueError):
            hud.pdf([1.0, 0.0])

    def test_sample(self):
        for dim in range(2, 5):
            hud = HypersphericalUniformDistribution(dim)
            n = 10
            samples = hud.sample(n)
            self.assertEqual(samples.shape, (n, hud.dim + 1))
            npt.assert_allclose(linalg.norm(samples, axis=1), ones(n), rtol=5e-7)

    def test_sample_accepts_numpy_integer_count(self):
        hud = HypersphericalUniformDistribution(2)

        samples = hud.sample(np.int64(4))

        self.assertEqual(samples.shape, (4, hud.dim + 1))
        npt.assert_allclose(linalg.norm(samples, axis=1), ones(4), rtol=5e-7)

    def test_sample_rejects_invalid_count(self):
        hud = HypersphericalUniformDistribution(2)

        for n in (0, -1, 2.5, True, [3], "3", b"3"):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    hud.sample(n)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Numerical mean direction not supported for this backend",
    )
    def test_mean_direction_numerical_rejects_uniform_full_sphere(self):
        """The full-sphere uniform distribution has no mean direction."""
        hud = HypersphericalUniformDistribution(1)

        with self.assertRaisesRegex(ValueError, "Mean direction is undefined"):
            hud.mean_direction_numerical()

    def test_ln_pdf(self):
        """Test if ln_pdf returns the correct logarithm of the probability density."""
        hud = HypersphericalUniformDistribution(3)
        n = 10
        samples = hud.sample(n)
        # Assert that the computed values are close to the expected values
        npt.assert_array_almost_equal(
            hud.ln_pdf(samples),
            log(hud.pdf(samples)),
            decimal=10,
            err_msg="ln_pdf does not return correct log probabilities.",
        )

    def test_ln_pdf_accepts_list_inputs(self):
        hud = HypersphericalUniformDistribution(2)
        points = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

        npt.assert_allclose(hud.ln_pdf(points), log(hud.pdf(array(points))))
        npt.assert_allclose(
            float(hud.ln_pdf(points[0])),
            math.log(float(hud.pdf(array(points[0])))),
        )

    def test_ln_pdf_rejects_wrong_dimension(self):
        hud = HypersphericalUniformDistribution(2)

        with self.assertRaises(ValueError):
            hud.ln_pdf([1.0, 0.0])

    def test_ln_pdf_single_point_matches_pdf(self):
        """Single-point ln_pdf should be scalar-like and match log(pdf)."""
        random.seed(1)
        for dim in range(2, 5):
            hud = HypersphericalUniformDistribution(dim)
            x = random.uniform(size=(dim + 1,))
            x = x / linalg.norm(x)
            npt.assert_allclose(hud.ln_pdf(x), log(hud.pdf(x)), atol=1e-12)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    def test_uniform_mean_direction_is_undefined(self):
        """The full-sphere uniform distribution has no mean direction."""
        hud = HypersphericalUniformDistribution(2)

        with self.assertWarnsRegex(UserWarning, "Mean direction is undefined"):
            with self.assertRaisesRegex(ValueError, "Mean direction is undefined"):
                hud.mean_direction_numerical()


if __name__ == "__main__":
    unittest.main()
