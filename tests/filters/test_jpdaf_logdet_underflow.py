import math
import unittest

import pyrecest.backend
from pyrecest.backend import array
from pyrecest.filters import JPDAF


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Only supported on numpy backend",
)
class JointProbabilisticDataAssociationFilterLogdetTest(unittest.TestCase):
    def test_log_gaussian_likelihood_handles_tiny_positive_definite_covariance(self):
        log_likelihood, mahalanobis_distance = JPDAF._log_gaussian_likelihood(
            array([0.0, 0.0]),
            array([[1e-200, 0.0], [0.0, 1e-200]]),
        )

        expected_log_likelihood = -math.log(2.0 * math.pi) + 200.0 * math.log(10.0)
        self.assertEqual(mahalanobis_distance, 0.0)
        self.assertTrue(math.isfinite(log_likelihood))
        self.assertAlmostEqual(log_likelihood, expected_log_likelihood)


if __name__ == "__main__":
    unittest.main()
