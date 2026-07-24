import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.sampling.euclidean_sampler import FibonacciGridSampler


class TestFibonacciGaussianParameterValidation(unittest.TestCase):
    def setUp(self):
        self.sampler = FibonacciGridSampler()

    def test_real_object_arrays_remain_supported(self):
        covariance = np.array([[2.0, 0.5], [0.5, 1.0]], dtype=object)
        mean = np.array([1.0, -1.0], dtype=object)

        samples = self.sampler.get_gaussian_samples(
            20, 2, covariance=covariance, mean=mean
        )

        self.assertEqual(samples.shape, (20, 2))
        self.assertTrue(np.all(np.isfinite(samples)))
        npt.assert_allclose(samples.mean(axis=0), np.asarray(mean, dtype=float))

    def test_complex_covariance_is_rejected_without_lossy_cast(self):
        invalid_covariances = (
            np.array([[1.0 + 1.0j, 0.0], [0.0, 1.0]]),
            np.array([[1.0 + 1.0j, 0.0], [0.0, 1.0]], dtype=object),
        )

        for covariance in invalid_covariances:
            with self.subTest(dtype=covariance.dtype):
                with self.assertRaisesRegex(ValueError, "covariance.*real numeric"):
                    self.sampler.get_gaussian_samples(
                        10, 2, covariance=covariance, mean=np.zeros(2)
                    )

    def test_complex_mean_is_rejected_without_lossy_cast(self):
        invalid_means = (
            np.array([0.0 + 1.0j, 0.0]),
            np.array([0.0 + 1.0j, 0.0], dtype=object),
        )

        for mean in invalid_means:
            with self.subTest(dtype=mean.dtype):
                with self.assertRaisesRegex(ValueError, "mean.*real numeric"):
                    self.sampler.get_gaussian_samples(10, 2, mean=mean)

    def test_nonvector_mean_shapes_are_rejected(self):
        invalid_means = (
            np.zeros((1, 2)),
            np.zeros((2, 1)),
            np.zeros((2, 1, 1)),
        )

        for mean in invalid_means:
            with self.subTest(shape=mean.shape):
                with self.assertRaisesRegex(ValueError, "mean must have shape"):
                    self.sampler.get_gaussian_samples(10, 2, mean=mean)


if __name__ == "__main__":
    unittest.main()
