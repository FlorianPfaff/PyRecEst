import unittest

import numpy as np
from pyrecest.backend import asarray
from pyrecest.sampling import JulierSigmaPoints, MerweScaledSigmaPoints


class TestSigmaPointMeanShapeValidation(unittest.TestCase):
    def test_sigma_points_reject_nonvector_means(self):
        point_sets = (
            JulierSigmaPoints(n=2, kappa=1.0),
            MerweScaledSigmaPoints(n=2, alpha=0.5, beta=2.0, kappa=0.0),
        )
        invalid_means = (
            np.array([[0.0, 1.0]]),
            np.array([[0.0], [1.0]]),
            np.array([[[0.0]], [[1.0]]]),
        )
        covariance = asarray(np.eye(2))

        for point_set in point_sets:
            for mean in invalid_means:
                with self.subTest(point_set=type(point_set).__name__, shape=mean.shape):
                    with self.assertRaisesRegex(
                        ValueError, r"x must have shape \(2,\)"
                    ):
                        point_set.sigma_points(asarray(mean), covariance)


if __name__ == "__main__":
    unittest.main()
