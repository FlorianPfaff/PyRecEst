import unittest

import numpy as np
from pyrecest.filters import WeightedGaussianHypothesis


class GaussianHypothesisCovarianceValidationTest(unittest.TestCase):
    def test_rejects_negative_scalar_variance(self):
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            WeightedGaussianHypothesis(
                mean=np.array([0.0]),
                covariance=np.array([[-1.0]]),
            )

    def test_rejects_indefinite_covariance_with_positive_diagonal(self):
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            WeightedGaussianHypothesis(
                mean=np.array([0.0, 0.0]),
                covariance=np.array([[1.0, 2.0], [2.0, 1.0]]),
            )

    def test_accepts_positive_semidefinite_covariance(self):
        hypothesis = WeightedGaussianHypothesis(
            mean=np.array([0.0, 0.0]),
            covariance=np.array([[1.0, 1.0], [1.0, 1.0]]),
        )

        self.assertTrue(np.allclose(hypothesis.covariance, np.ones((2, 2))))


if __name__ == "__main__":
    unittest.main()
