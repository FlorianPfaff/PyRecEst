import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.backend import mean, ones, random, std, zeros
from pyrecest.sampling.euclidean_sampler import FibonacciGridSampler, GaussianSampler
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

    def test_sample_mean_close_to_zero(self):
        """Check that the mean is close to 0 for each dimension."""
        means = mean(self.samples, axis=0)
        npt.assert_allclose(means, zeros(self.dim), atol=0.2)

    def test_sample_std_dev_close_to_one(self):
        """Check that the standard deviation is close to 1 for each dimension."""
        std_devs = std(self.samples, axis=0)
        npt.assert_allclose(std_devs, ones(self.dim), atol=0.2)

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


if __name__ == "__main__":
    unittest.main()
