import unittest

import numpy.testing as npt
import pyrecest.backend
from parameterized import parameterized

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, linalg, ones, sqrt
from pyrecest.distributions import VonMisesFisherDistribution
from pyrecest.distributions.hypersphere_subset.hyperspherical_dirac_distribution import (
    HypersphericalDiracDistribution,
)

vectors_to_test_2d = array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        array([1.0, 1.0, 0.0]) / sqrt(2.0),
        array([1.0, 1.0, 2.0]) / linalg.norm(array([1.0, 1.0, 2.0])),
        -array([1.0, 1.0, 2.0]) / linalg.norm(array([1.0, 1.0, 2.0])),
    ]
)


class TestVonMisesFisherDistribution(unittest.TestCase):
    def setUp(self):
        self.mu = array([1.0, 2.0, 3.0])
        self.mu = self.mu / linalg.norm(self.mu)
        self.kappa = 2
        self.vmf = VonMisesFisherDistribution(self.mu, self.kappa)
        self.other = VonMisesFisherDistribution(
            array([0.0, 0.0, 1.0]), self.kappa / 3.0
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    def test_integrate_2d(self):
        self.assertAlmostEqual(self.vmf.integrate(), 1.0, delta=1e-6)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    def test_integrate_3d(self):
        mu_s3 = array([1.0, 2.0, 3.0, 4.0]) / linalg.norm(array([1.0, 2.0, 3.0, 4.0]))
        self.assertAlmostEqual(
            VonMisesFisherDistribution(mu_s3, self.kappa).integrate(), 1.0, delta=1e-6
        )

    def test_vmf_distribution_3d_sanity_check(self):
        self.assertIsInstance(self.vmf, VonMisesFisherDistribution)
        self.assertTrue(allclose(self.vmf.mu, self.mu))
        self.assertEqual(self.vmf.kappa, self.kappa)
        self.assertEqual(self.vmf.dim + 1, len(self.mu))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    def test_vmf_distribution_3d_mode(self):
        npt.assert_allclose(self.vmf.mode_numerical(), self.vmf.mode(), atol=1e-5)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    def test_vmf_distribution_3d_integral(self):
        self.assertAlmostEqual(self.vmf.integrate(), 1, delta=1e-5)

    def test_vmf_distribution_3d_multiplication(self):
        vmf_mul = self.vmf.multiply(self.other)
        vmf_mul2 = self.other.multiply(self.vmf)
        c = vmf_mul.pdf(array([1.0, 0.0, 0.0])) / (
            self.vmf.pdf(array([1.0, 0.0, 0.0]))
            * self.other.pdf(array([1.0, 0.0, 0.0]))
        )
        x = array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(
            self.vmf.pdf(x) * self.other.pdf(x) * c, vmf_mul.pdf(x), delta=1e-10
        )
        self.assertAlmostEqual(
            self.vmf.pdf(x) * self.other.pdf(x) * c, vmf_mul2.pdf(x), delta=1e-10
        )

    def test_vmf_distribution_3d_convolve(self):
        vmf_conv = self.vmf.convolve(self.other)
        self.assertTrue(allclose(vmf_conv.mu, self.vmf.mu, atol=1e-10))
        d = 3
        self.assertAlmostEqual(
            VonMisesFisherDistribution.a_d(d, vmf_conv.kappa),
            VonMisesFisherDistribution.a_d(d, self.vmf.kappa)
            * VonMisesFisherDistribution.a_d(d, self.other.kappa),
            delta=1e-10,
        )

    def test_init_2d(self):
        mu = array([1.0, 1.0, 2.0])
        mu = mu / linalg.norm(mu)
        kappa = 10.0
        dist = VonMisesFisherDistribution(mu, kappa)
        npt.assert_array_almost_equal(dist.C, 7.22562325261744e-05)

    def test_init_3d(self):
        mu = array([1.0, 1.0, 2.0, -3.0])
        mu = mu / linalg.norm(mu)
        kappa = 2.0
        dist = VonMisesFisherDistribution(mu, kappa)
        npt.assert_array_almost_equal(dist.C, 0.0318492506152322)

    def test_pdf_2d(self):
        mu = array([1.0, 1.0, 2.0])
        mu = mu / linalg.norm(mu)
        kappa = 10.0
        dist = VonMisesFisherDistribution(mu, kappa)

        npt.assert_array_almost_equal(
            dist.pdf(vectors_to_test_2d),
            array(
                [
                    0.00428425301914546,
                    0.00428425301914546,
                    0.254024093013817,
                    0.0232421165060131,
                    1.59154943419939,
                    3.28042788159008e-09,
                ],
            ),
        )

    def test_pdf_3d(self):
        mu = array([1.0, 1.0, 2.0, -3.0])
        mu = mu / linalg.norm(mu)
        kappa = 2.0
        dist = VonMisesFisherDistribution(mu, kappa)

        xs_unnorm = array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, -1.0],
            ]
        )
        xs = xs_unnorm / linalg.norm(xs_unnorm, axis=1).reshape(-1, 1)

        npt.assert_array_almost_equal(
            dist.pdf(xs),
            array(
                [
                    0.0533786916025838,
                    0.0533786916025838,
                    0.0894615936690536,
                    0.00676539409350726,
                    0.0661093769275549,
                    0.0318492506152322,
                    0.0952456142366906,
                    0.0221063629087443,
                    0.0153438863274034,
                    0.13722300001807,
                ],
            ),
        )

    def test_mean_direction(self):
        mu = 1.0 / sqrt(2.0) * array([1.0, 1.0, 0.0])
        vmf = VonMisesFisherDistribution(mu, 1)
        self.assertTrue(allclose(vmf.mean_direction(), mu, atol=1e-13))

    def _test_hellinger_distance_helper(
        self, dist1, dist2, delta=1e-10, numerical_delta=1e-10
    ):
        self.assertAlmostEqual(dist1.hellinger_distance(dist1), 0.0, delta=delta)
        self.assertAlmostEqual(dist2.hellinger_distance(dist2), 0.0, delta=delta)
        self.assertAlmostEqual(
            dist1.hellinger_distance(dist2),
            dist1.hellinger_distance_numerical(dist2),
            delta=numerical_delta,
        )
        self.assertAlmostEqual(
            dist1.hellinger_distance(dist2),
            dist2.hellinger_distance(dist1),
            delta=delta,
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    def test_hellinger_distance_s2(self):
        self._test_hellinger_distance_helper(self.vmf, self.other)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    def test_hellinger_distance_s3(self):
        vmf1 = VonMisesFisherDistribution(array([1.0, 0.0, 0.0, 0.0]), array(0.6))
        mu2 = array([1.0, 2.0, 3.0, 4.0])
        vmf2 = VonMisesFisherDistribution(mu2 / linalg.norm(mu2), array(2.1))
        self._test_hellinger_distance_helper(vmf1, vmf2, numerical_delta=1e-6)

    @parameterized.expand(
        [
            ("2D_case", array([-1.0, 0.0, 0.0]), 1.3),
            ("3D_case", array([0.0, 1.0, 0.0, 0.0]), 0.5),
        ]
    )
    def test_from_distribution_vmf(self, _, mu, kappa):
        vmf1 = VonMisesFisherDistribution(mu, kappa)
        vmf2 = VonMisesFisherDistribution.from_distribution(vmf1)

        npt.assert_allclose(vmf1.mu, vmf2.mu, rtol=1e-10)
        npt.assert_allclose(vmf1.kappa, vmf2.kappa, rtol=5e-7)

    def test_from_distribution_dirac(self):
        dirac_dist = HypersphericalDiracDistribution(
            array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    array([0.0, 1.0, 1.0]) / linalg.norm(array([0.0, 1.0, 1.0])),
                ]
            ).T
        )
        vmf = VonMisesFisherDistribution.from_distribution(dirac_dist)

        npt.assert_allclose(dirac_dist.mean(), vmf.mean())

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    @parameterized.expand(
        [
            ("s2", array([1.0, 2.0, 3.0]), 2.0),
            ("s3", array([1.0, 2.0, 3.0, 4.0]), 0.5),
        ]
    )
    def test_sample_deterministic_matches_mean_resultant_vector(self, _, mu, kappa):
        mu = mu / linalg.norm(mu)
        vmf = VonMisesFisherDistribution(mu, kappa)

        samples = vmf.sample_deterministic()

        expected_n_samples = 2 * vmf.dim + 1
        weights = ones(expected_n_samples) / expected_n_samples
        self.assertEqual(samples.shape, (vmf.input_dim, expected_n_samples))
        npt.assert_allclose(
            linalg.norm(samples, axis=0), ones(expected_n_samples), atol=1e-12
        )
        npt.assert_allclose(
            samples @ weights, vmf.mean_resultant_vector(), atol=1e-12
        )

    def test_a_d_uses_scaled_bessel_ratio_for_large_kappa(self):
        npt.assert_allclose(VonMisesFisherDistribution.a_d(3, 1000.0), 0.999)


if __name__ == "__main__":
    unittest.main()
