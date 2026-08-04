import unittest

import numpy as np
from pyrecest.utils.metrics import (
    extent_wasserstein_distance,
    gaussian_wasserstein_distance,
)


class TestWassersteinCovarianceSymmetryValidation(unittest.TestCase):
    def test_gaussian_wasserstein_rejects_asymmetric_covariances(self):
        mean = np.zeros(2)
        valid = np.eye(2)
        asymmetric = np.array([[1.0, 3.0], [-3.0, 1.0]])

        with self.assertRaisesRegex(ValueError, "covariance1 must be symmetric"):
            gaussian_wasserstein_distance(mean, asymmetric, mean, valid)
        with self.assertRaisesRegex(ValueError, "covariance2 must be symmetric"):
            gaussian_wasserstein_distance(mean, valid, mean, asymmetric)

    def test_extent_wasserstein_rejects_asymmetric_extents(self):
        valid = np.eye(2)
        asymmetric = np.array([[1.0, 3.0], [-3.0, 1.0]])

        with self.assertRaisesRegex(ValueError, "estimated_extent must be symmetric"):
            extent_wasserstein_distance(asymmetric, valid)
        with self.assertRaisesRegex(ValueError, "reference_extent must be symmetric"):
            extent_wasserstein_distance(valid, asymmetric)

    def test_roundoff_skew_remains_accepted(self):
        mean = np.zeros(2)
        valid = np.eye(2)
        roundoff_skew = np.array([[1.0, 5e-13], [-5e-13, 1.0]])

        self.assertEqual(
            gaussian_wasserstein_distance(mean, roundoff_skew, mean, valid),
            0.0,
        )

    def test_extreme_asymmetry_raises_targeted_error_under_strict_numpy(self):
        mean = np.zeros(2)
        valid = np.eye(2)
        max_float = np.finfo(np.float64).max
        asymmetric = np.array([[1.0, max_float], [-max_float, 1.0]])

        with np.errstate(all="raise"):
            with self.assertRaisesRegex(ValueError, "covariance1 must be symmetric"):
                gaussian_wasserstein_distance(mean, asymmetric, mean, valid)


if __name__ == "__main__":
    unittest.main()
