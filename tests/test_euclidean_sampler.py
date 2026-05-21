import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.backend import mean, ones, random, std, zeros
from pyrecest.sampling import (
    FibonacciRejectionSampler as PublicFibonacciRejectionSampler,
)
from pyrecest.sampling import HaltonGridSampler as PublicHaltonGridSampler
from pyrecest.sampling import SobolGridSampler as PublicSobolGridSampler
from pyrecest.sampling.euclidean_sampler import (
    FibonacciGridSampler,
    FibonacciRejectionSampler,
    GaussianSampler,
    HaltonGridSampler,
    SobolGridSampler,
)
from scipy.stats import shapiro


class TestGaussianSampler(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        self.sampler = GaussianSampler()
        self.n_samples = 200
        self.dim = 2
        self.samples = self.sampler.sample_stochastic(self.n_samples, self.dim)

    def test_sample_stochastic_shape(self):
        """Check that the returned shape matches the requested number of samples and dimension."""
        self.assertEqual(self.samples.shape, (self.n_samples, self.dim))

    def test_sample_stochastic_reproducible_after_reseed(self):
        """Repeated reseeding must reproduce identical Gaussian samples."""
        random.seed(0)
        samples1 = self.sampler.sample_stochastic(self.n_samples, self.dim)
        random.seed(0)
        samples2 = self.sampler.sample_stochastic(self.n_samples, self.dim)
        npt.assert_allclose(samples1, samples2)

    def test_sample_mean_close_to_zero(self):
        """Check that the mean is close to 0 for each dimension."""
        means = mean(self.samples, axis=0)
        npt.assert_allclose(means, zeros(self.dim), atol=0.2)

    def test_sample_std_dev_close_to_one(self):
        """Check that the standard deviation is close to 1 for each dimension."""
        std_devs = std(self.samples, axis=0)
        npt.assert_allclose(std_devs, ones(self.dim), atol=0.15)

    def test_samples_follow_gaussian_distribution(self):
        """Test if the samples follow a Gaussian distribution for each dimension."""
        for i in range(self.dim):
            _, p_value = shapiro(self.samples[:, i])
            self.assertGreater(p_value, 0.05)


class TestFibonacciGridSampler(unittest.TestCase):
    def setUp(self):
        self.sampler = FibonacciGridSampler()

    def test_sample_stochastic_shape_2d(self):
        """Shape must be (n_samples, dim) for a 2-D grid."""
        samples = self.sampler.sample_stochastic(100, 2)
        self.assertEqual(samples.shape, (100, 2))

    def test_sample_stochastic_shape_3d(self):
        """Shape must be (n_samples, dim) for a 3-D grid."""
        samples = self.sampler.sample_stochastic(50, 3)
        self.assertEqual(samples.shape, (50, 3))

    def test_sample_stochastic_singleton_is_finite_origin(self):
        """A singleton moment-matched Gaussian grid should be finite and centered."""
        samples = self.sampler.sample_stochastic(1, 2)
        self.assertEqual(samples.shape, (1, 2))
        self.assertTrue(np.all(np.isfinite(samples)))
        npt.assert_allclose(samples, np.zeros((1, 2)))

    def test_get_gaussian_samples_singleton_returns_requested_mean(self):
        """A singleton transformed Gaussian grid should represent the requested mean."""
        mu = np.array([3.0, -2.0])
        cov = np.array([[2.0, 0.5], [0.5, 1.0]])
        samples = self.sampler.get_gaussian_samples(1, 2, covariance=cov, mean=mu)
        self.assertEqual(samples.shape, (1, 2))
        self.assertTrue(np.all(np.isfinite(samples)))
        npt.assert_allclose(samples[0], mu)

    def test_get_uniform_samples_singleton_is_center(self):
        """A singleton uniform Fibonacci grid should be the center of the unit cube."""
        samples = self.sampler.get_uniform_samples(1, 3)
        self.assertEqual(samples.shape, (1, 3))
        npt.assert_allclose(samples, np.full((1, 3), 0.5))

    def test_sample_stochastic_mean_close_to_zero(self):
        """Marginal means of the moment-matched samples should be ~0."""
        samples = self.sampler.sample_stochastic(200, 2)
        npt.assert_allclose(samples.mean(axis=0), np.zeros(2), atol=1e-10)

    def test_sample_stochastic_std_close_to_one(self):
        """Marginal standard deviations should be ~1 after moment matching."""
        samples = self.sampler.sample_stochastic(200, 2)
        npt.assert_allclose(samples.std(axis=0, ddof=0), np.ones(2), atol=1e-10)

    def test_sample_stochastic_deterministic(self):
        """Calling sample_stochastic twice should return identical arrays."""
        s1 = self.sampler.sample_stochastic(50, 2)
        s2 = self.sampler.sample_stochastic(50, 2)
        npt.assert_array_equal(s1, s2)

    def test_get_uniform_samples_range(self):
        """Uniform samples must lie in [0, 1]^d."""
        samples = self.sampler.get_uniform_samples(100, 2)
        self.assertEqual(samples.shape, (100, 2))
        self.assertTrue(np.all(samples >= 0.0))
        self.assertTrue(np.all(samples <= 1.0))

    def test_get_gaussian_samples_shape(self):
        """Gaussian samples must have the correct shape."""
        cov = np.array([[2.0, 0.5], [0.5, 1.0]])
        mu = np.array([1.0, -1.0])
        samples = self.sampler.get_gaussian_samples(100, 2, covariance=cov, mean=mu)
        self.assertEqual(samples.shape, (100, 2))

    def test_get_gaussian_samples_mean(self):
        """Sample mean should be close to the requested mean."""
        mu = np.array([3.0, -2.0])
        samples = self.sampler.get_gaussian_samples(200, 2, mean=mu)
        npt.assert_allclose(samples.mean(axis=0), mu, atol=0.5)

    def test_zero_samples(self):
        """Requesting zero samples should return an empty (0, dim) array."""
        samples = self.sampler.sample_stochastic(0, 2)
        self.assertEqual(samples.shape, (0, 2))

    def test_invalid_grid_arguments(self):
        """Fibonacci grid generation should reject invalid dimensions and counts."""
        with self.assertRaises(ValueError):
            self.sampler.sample_stochastic(-1, 2)
        with self.assertRaises(ValueError):
            self.sampler.sample_stochastic(1, 0)

    def test_fibonacci_eigen_d4(self):
        """fibonacci_eigen for D=4 should return orthonormal V."""
        from pyrecest.sampling.euclidean_sampler import _fibonacci_eigen

        V, R = _fibonacci_eigen(4)
        self.assertEqual(V.shape, (4, 4))
        self.assertEqual(R.shape, (4,))
        npt.assert_allclose(V.T @ V, np.eye(4), atol=1e-12)

    def test_fibonacci_eigen_general(self):
        """fibonacci_eigen for a prime-based dimension should return orthonormal V."""
        from pyrecest.sampling.euclidean_sampler import _fibonacci_eigen

        for d in (1, 2, 3, 5):
            V, R = _fibonacci_eigen(d)
            self.assertEqual(V.shape, (d, d))
            self.assertEqual(R.shape, (d,))
            npt.assert_allclose(V.T @ V, np.eye(d), atol=1e-12)


class TestFibonacciRejectionSampler(unittest.TestCase):
    def setUp(self):
        self.sampler = FibonacciRejectionSampler()

    def test_unit_density_accepts_all_unit_cube_candidates(self):
        """A unit-density target with max_density=1 should accept all candidates."""
        n_candidates = 25
        samples, info = self.sampler.sample_rejection(
            lambda xs: np.ones(xs.shape[0]),
            n_candidates=n_candidates,
            dim=2,
            max_density=1.0,
        )
        expected = FibonacciGridSampler().get_uniform_samples(n_candidates, 3)[:, :2]

        self.assertEqual(samples.shape, (n_candidates, 2))
        npt.assert_allclose(samples, expected)
        self.assertEqual(info["n_candidates"], n_candidates)
        self.assertEqual(info["n_accepted"], n_candidates)
        self.assertEqual(info["n_rejected"], 0)
        self.assertEqual(info["acceptance_rate"], 1.0)
        npt.assert_allclose(info["bounding_box"], np.array([[0.0, 1.0], [0.0, 1.0]]))

    def test_bounding_box_mapping(self):
        """Accepted samples should be mapped from the unit grid into the bounding box."""
        n_candidates = 30
        bounding_box = np.array([[-2.0, 2.0], [10.0, 20.0]])
        samples, _ = self.sampler.sample_rejection(
            lambda xs: np.ones(xs.shape[0]),
            n_candidates=n_candidates,
            dim=2,
            max_density=1.0,
            bounding_box=bounding_box,
        )
        unit_samples = FibonacciGridSampler().get_uniform_samples(n_candidates, 3)[
            :, :2
        ]
        expected = unit_samples * np.array([4.0, 10.0]) + np.array([-2.0, 10.0])

        npt.assert_allclose(samples, expected)
        self.assertTrue(np.all(samples[:, 0] >= -2.0))
        self.assertTrue(np.all(samples[:, 0] <= 2.0))
        self.assertTrue(np.all(samples[:, 1] >= 10.0))
        self.assertTrue(np.all(samples[:, 1] <= 20.0))

    def test_rejection_rule_matches_fibonacci_ordinate(self):
        """The last Fibonacci-grid coordinate should drive deterministic rejection."""
        n_candidates = 40
        samples, info = self.sampler.sample_rejection(
            lambda xs: np.full(xs.shape[0], 0.5),
            n_candidates=n_candidates,
            dim=1,
            max_density=1.0,
        )
        proposal_grid = FibonacciGridSampler().get_uniform_samples(n_candidates, 2)
        expected = proposal_grid[proposal_grid[:, 1] <= 0.5, :1]

        npt.assert_allclose(samples, expected)
        self.assertEqual(info["n_accepted"], expected.shape[0])
        self.assertEqual(info["n_rejected"], n_candidates - expected.shape[0])

    def test_sampling_is_deterministic(self):
        """Repeated calls with the same density should return identical accepted samples."""

        def pdf(xs):
            return np.exp(-np.sum(xs**2, axis=1))

        bounding_box = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        samples1, info1 = self.sampler.sample_rejection(
            pdf, 60, 2, 1.0, bounding_box=bounding_box
        )
        samples2, info2 = self.sampler.sample_rejection(
            pdf, 60, 2, 1.0, bounding_box=bounding_box
        )

        npt.assert_array_equal(samples1, samples2)
        self.assertEqual(info1["n_accepted"], info2["n_accepted"])
        self.assertEqual(info1["n_rejected"], info2["n_rejected"])

    def test_zero_candidates(self):
        """Requesting no candidates should return empty samples and zero metadata."""
        samples, info = self.sampler.sample_rejection(
            lambda xs: np.ones(xs.shape[0]),
            n_candidates=0,
            dim=2,
            max_density=1.0,
        )

        self.assertEqual(samples.shape, (0, 2))
        self.assertEqual(info["n_candidates"], 0)
        self.assertEqual(info["n_accepted"], 0)
        self.assertEqual(info["n_rejected"], 0)
        self.assertEqual(info["acceptance_rate"], 0.0)

    def test_max_density_must_bound_pdf(self):
        """The sampler should reject invalid density upper bounds."""
        with self.assertRaises(ValueError):
            self.sampler.sample_rejection(
                lambda xs: np.ones(xs.shape[0]),
                n_candidates=10,
                dim=1,
                max_density=0.5,
            )

    def test_invalid_bounding_box(self):
        """Bounding boxes need one lower and one upper value per dimension."""
        with self.assertRaises(ValueError):
            self.sampler.sample_rejection(
                lambda xs: np.ones(xs.shape[0]),
                n_candidates=10,
                dim=2,
                max_density=1.0,
                bounding_box=np.array([[0.0, 1.0]]),
            )

    def test_sample_stochastic_is_not_available(self):
        """Density-free stochastic sampling is not the rejection sampler interface."""
        with self.assertRaises(NotImplementedError):
            self.sampler.sample_stochastic(10, 2)

    def test_public_sampling_import(self):
        """FibonacciRejectionSampler should be exported from pyrecest.sampling."""
        self.assertIs(PublicFibonacciRejectionSampler, FibonacciRejectionSampler)


class TestDeterministicProposalGridSamplers(unittest.TestCase):
    def test_sobol_first_points(self):
        """Sobol samples should match the unscrambled low-discrepancy sequence."""
        samples = SobolGridSampler().get_uniform_samples(4, 2)
        expected = np.array(
            [
                [0.0, 0.0],
                [0.5, 0.5],
                [0.75, 0.25],
                [0.25, 0.75],
            ]
        )
        npt.assert_allclose(samples, expected)

    def test_halton_first_points(self):
        """Halton samples should match the unscrambled low-discrepancy sequence."""
        samples = HaltonGridSampler().get_uniform_samples(5, 2)
        expected = np.array(
            [
                [0.0, 0.0],
                [0.5, 1.0 / 3.0],
                [0.25, 2.0 / 3.0],
                [0.75, 1.0 / 9.0],
                [0.125, 4.0 / 9.0],
            ]
        )
        npt.assert_allclose(samples, expected)

    def test_proposal_grids_are_deterministic(self):
        """Repeated calls should return identical proposal grids."""
        for sampler in (SobolGridSampler(), HaltonGridSampler()):
            with self.subTest(sampler=sampler.__class__.__name__):
                samples1 = sampler.get_uniform_samples(16, 3)
                samples2 = sampler.get_uniform_samples(16, 3)
                npt.assert_array_equal(samples1, samples2)

    def test_sample_stochastic_returns_unit_hypercube_proposals(self):
        """The sampler interface should return deterministic unit-hypercube points."""
        for sampler in (SobolGridSampler(), HaltonGridSampler()):
            with self.subTest(sampler=sampler.__class__.__name__):
                samples = sampler.sample_stochastic(8, 3)
                self.assertEqual(samples.shape, (8, 3))
                self.assertTrue(np.all(samples >= 0.0))
                self.assertTrue(np.all(samples < 1.0))

    def test_zero_samples(self):
        """Requesting zero proposals should return an empty two-dimensional array."""
        for sampler in (SobolGridSampler(), HaltonGridSampler()):
            with self.subTest(sampler=sampler.__class__.__name__):
                samples = sampler.get_uniform_samples(0, 3)
                self.assertEqual(samples.shape, (0, 3))

    def test_public_sampling_imports(self):
        """Sobol and Halton samplers should be exported from pyrecest.sampling."""
        self.assertIs(PublicSobolGridSampler, SobolGridSampler)
        self.assertIs(PublicHaltonGridSampler, HaltonGridSampler)


if __name__ == "__main__":
    unittest.main()
