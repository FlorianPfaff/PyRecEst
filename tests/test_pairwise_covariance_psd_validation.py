import unittest

from pyrecest.backend import array, zeros
from pyrecest.utils import (
    pairwise_covariance_shape_components,
    pairwise_mahalanobis_distances,
)


class TestPairwiseCovariancePSDValidation(unittest.TestCase):
    @staticmethod
    def _indefinite_covariance():
        return array(
            [
                [[-2.0], [0.0]],
                [[0.0], [1.0]],
            ]
        )

    def test_mahalanobis_rejects_indefinite_covariance(self):
        means_a = array([[0.0], [0.0]])
        means_b = array([[1.0], [0.0]])

        with self.assertRaisesRegex(ValueError, "positive-semidefinite"):
            pairwise_mahalanobis_distances(
                means_a,
                self._indefinite_covariance(),
                means_b,
                zeros((2, 2, 1)),
            )

    def test_shape_components_reject_indefinite_covariance(self):
        with self.assertRaisesRegex(ValueError, "positive-semidefinite"):
            pairwise_covariance_shape_components(
                self._indefinite_covariance(),
                zeros((2, 2, 1)),
            )

    def test_roundoff_scale_negative_eigenvalue_is_tolerated(self):
        near_psd_covariance = array(
            [
                [[1.0], [0.0]],
                [[0.0], [-1.0e-12]],
            ]
        )

        distances = pairwise_mahalanobis_distances(
            array([[0.0], [0.0]]),
            near_psd_covariance,
            array([[1.0], [0.0]]),
            zeros((2, 2, 1)),
            regularization=1.0,
        )

        self.assertEqual(distances.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
