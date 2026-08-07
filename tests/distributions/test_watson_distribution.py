import unittest

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, linalg, log
from pyrecest.distributions import (
    BinghamDistribution,
    HypersphericalUniformDistribution,
    WatsonDistribution,
)


class TestWatsonDistribution(unittest.TestCase):
    def setUp(self):
        self.xs = array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 2.0, 2.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, -1.0, -1.0],
            ],
            dtype=float,
        )
        self.xs = self.xs / linalg.norm(self.xs, axis=1).reshape((-1, 1))

    def test_constructor(self):
        mu = array([1.0, 2.0, 3.0])
        mu = mu / linalg.norm(mu)
        kappa = 2.0
        w = WatsonDistribution(mu, kappa)

        self.assertIsInstance(w, WatsonDistribution)
        npt.assert_array_equal(w.mu, mu)
        self.assertEqual(w.kappa, kappa)
        self.assertEqual(w.input_dim, mu.shape[0])

    def test_constructor_accepts_list_mu(self):
        mu = array([1.0, 2.0, 3.0])
        mu = mu / linalg.norm(mu)
        w = WatsonDistribution([float(v) for v in mu], 2.0)

        npt.assert_allclose(w.mu, mu)

    def test_constructor_rejects_invalid_parameters(self):
        invalid_cases = [
            ([[1.0, 0.0, 0.0]], 2.0),
            ([1.0], 2.0),
            ([1.0, 1.0, 0.0], 2.0),
            ([float("nan"), 0.0, 1.0], 2.0),
            ([1.0, 0.0, 0.0], float("nan")),
            ([1.0, 0.0, 0.0], float("inf")),
            ([1.0, 0.0, 0.0], [1.0, 2.0]),
        ]

        for mu, kappa in invalid_cases:
            with self.subTest(mu=mu, kappa=kappa):
                with self.assertRaises(ValueError):
                    WatsonDistribution(mu, kappa)

    def test_pdf(self):
        mu = array([1.0, 2.0, 3.0])
        mu = mu / linalg.norm(mu)
        kappa = 2.0
        w = WatsonDistribution(mu, kappa)

        expected_pdf_values = array(
            [
                0.0388240901641662,
                0.229710245437696,
                0.0595974246790006,
                0.121741272709942,
                0.186880524436683,
                0.186880524436683,
            ]
        )

        pdf_values = w.pdf(self.xs)
        npt.assert_array_almost_equal(pdf_values, expected_pdf_values, decimal=5)

    def test_pdf_stays_finite_for_large_positive_kappa(self):
        mu = array([1.0, 0.0, 0.0])
        dist = WatsonDistribution(mu, 1000.0)

        npt.assert_allclose(dist.pdf(mu), 159.07528584318962, rtol=1e-6)

    def test_pdf_accepts_list_inputs(self):
        mu = array([1.0, 2.0, 3.0])
        mu = mu / linalg.norm(mu)
        w = WatsonDistribution(mu, 2.0)
        xs = [[float(v) for v in self.xs[0]], [float(v) for v in self.xs[1]]]

        npt.assert_allclose(w.pdf(xs), w.pdf(array(xs)))
        npt.assert_allclose(w.pdf(xs[0]), w.pdf(array(xs[0])))

    def test_pdf_rejects_wrong_dimension(self):
        dist = WatsonDistribution(array([1.0, 0.0, 0.0]), 2.0)

        with self.assertRaises(ValueError):
            dist.pdf([1.0, 0.0])

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    def test_integrate(self):
        mu = array([1.0, 2.0, 3.0])
        mu = mu / linalg.norm(mu)
        kappa = 2.0
        w = WatsonDistribution(mu, kappa)
        self.assertAlmostEqual(w.integrate(), 1, delta=1e-5)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    def test_to_bingham(self):
        mu = array([1.0, 0.0, 0.0])
        kappa = 2.0
        watson_dist = WatsonDistribution(mu, kappa)
        bingham_dist = watson_dist.to_bingham()
        self.assertIsInstance(bingham_dist, BinghamDistribution)
        npt.assert_array_almost_equal(
            watson_dist.pdf(self.xs), bingham_dist.pdf(self.xs), decimal=5
        )

    def test_ln_norm_const(self):
        # Create an instance of WatsonDistribution
        # Here mu is a normalized vector, let's use [1, 0, 0] for simplicity in 3D
        mu = array([1.0, 0.0, 0.0])
        kappa = 3.0  # Arbitrary concentration parameter

        # Instantiate WatsonDistribution
        watson_dist = WatsonDistribution(mu, kappa)

        # Calculate norm_const and ln_norm_const
        norm_const = watson_dist.norm_const
        ln_norm_const = watson_dist.ln_norm_const

        # Check if ln_norm_const is the ln of norm_const
        expected_ln_norm_const = log(norm_const)

        # Use allclose to compare the floating-point results within some tolerance
        npt.assert_allclose(ln_norm_const, expected_ln_norm_const, rtol=1e-6)

    def test_ln_pdf(self):
        """Test if ln_pdf returns the correct logarithm of the probability density."""
        mu = array([1.0, 0.0, 0.0])
        kappa = 2.0
        dist = WatsonDistribution(mu, kappa)

        n = 10
        samples = HypersphericalUniformDistribution(2).sample(n)
        # Assert that the computed values are close to the expected values
        npt.assert_allclose(
            dist.ln_pdf(samples),
            log(dist.pdf(samples)),
            rtol=1e-6,
            err_msg="ln_pdf does not return correct log probabilities.",
        )

    def test_ln_pdf_rejects_wrong_dimension(self):
        dist = WatsonDistribution(array([1.0, 0.0, 0.0]), 2.0)

        with self.assertRaises(ValueError):
            dist.ln_pdf([1.0, 0.0])

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "JAX proposals use an explicit PRNG-key signature",
    )
    def test_metropolis_hastings_custom_proposal_uses_default_start_point(self):
        mu = array([1.0, 0.0, 0.0])
        dist = WatsonDistribution(mu, 2.0)

        def proposal(_):
            return array([[1.0, 0.0, 0.0]])

        samples = dist.sample_metropolis_hastings(
            3,
            burn_in=0,
            skipping=1,
            proposal=proposal,
        )

        self.assertEqual(samples.shape, (3, 3))

    def test_set_mode_returns_new_distribution(self):
        mu = array([0.0, 0.0, 1.0])
        new_mode = array([1.0, 0.0, 0.0])
        dist = WatsonDistribution(mu, 2.0)

        shifted = dist.set_mode(new_mode)

        self.assertIsNot(shifted, dist)
        npt.assert_allclose(dist.mu, mu)
        npt.assert_allclose(shifted.mu, new_mode)

    def test_set_mode_rejects_invalid_direction(self):
        dist = WatsonDistribution(array([0.0, 0.0, 1.0]), 2.0)

        for new_mode in ([1.0, 0.0], [2.0, 0.0, 0.0]):
            with self.subTest(new_mode=new_mode):
                with self.assertRaises(ValueError):
                    dist.set_mode(new_mode)

    def test_shift_accepts_flat_canonical_mu_and_returns_new_distribution(self):
        mu = array([0.0, 0.0, 1.0])
        new_mode = array([1.0, 0.0, 0.0])
        dist = WatsonDistribution(mu, 2.0)

        shifted = dist.shift(new_mode)

        self.assertIsNot(shifted, dist)
        npt.assert_allclose(dist.mu, mu)
        npt.assert_allclose(shifted.mu, new_mode)

    def test_shift_rejects_noncanonical_base_direction(self):
        dist = WatsonDistribution(array([1.0, 0.0, 0.0]), 2.0)

        with self.assertRaisesRegex(ValueError, "compatibility"):
            dist.shift([0.0, 0.0, 1.0])

    def test_shift_accepts_list_mode(self):
        mu = array([0.0, 0.0, 1.0])
        new_mode = array([1.0, 0.0, 0.0])
        dist = WatsonDistribution([0.0, 0.0, 1.0], 2.0)

        shifted = dist.shift([1.0, 0.0, 0.0])

        self.assertIsNot(shifted, dist)
        npt.assert_allclose(dist.mu, mu)
        npt.assert_allclose(shifted.mu, new_mode)


# Running the tests
if __name__ == "__main__":
    unittest.main()
