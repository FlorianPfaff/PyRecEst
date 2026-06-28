import unittest

import numpy as np
from pyrecest.utils.metrics import (
    extent_wasserstein_distance,
    gaussian_wasserstein_distance,
)


class TestWassersteinCovarianceValidation(unittest.TestCase):
    def test_gaussian_wasserstein_rejects_indefinite_covariances(self):
        mean = np.zeros(2)
        valid = np.eye(2)
        indefinite = np.diag([1.0, -1.0])

        with self.assertRaisesRegex(
            ValueError, "covariance1 must be positive semidefinite"
        ):
            gaussian_wasserstein_distance(mean, indefinite, mean, valid)
        with self.assertRaisesRegex(
            ValueError, "covariance2 must be positive semidefinite"
        ):
            gaussian_wasserstein_distance(mean, valid, mean, indefinite)

    def test_extent_wasserstein_rejects_indefinite_extents(self):
        valid = np.eye(2)
        indefinite = np.diag([1.0, -1.0])

        with self.assertRaisesRegex(
            ValueError, "estimated_extent must be positive semidefinite"
        ):
            extent_wasserstein_distance(indefinite, valid)
        with self.assertRaisesRegex(
            ValueError, "reference_extent must be positive semidefinite"
        ):
            extent_wasserstein_distance(valid, indefinite)


if __name__ == "__main__":
    unittest.main()
