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

    def test_rejects_extreme_asymmetry_without_silent_symmetrization(self):
        maximum = np.finfo(float).max
        covariance = np.array([[1.0, maximum], [-maximum, 1.0]])
        previous_settings = np.seterr(all="raise")
        try:
            with self.assertRaisesRegex(ValueError, "covariance must be symmetric"):
                WeightedGaussianHypothesis(
                    mean=np.array([0.0, 0.0]),
                    covariance=covariance,
                )
        finally:
            np.seterr(**previous_settings)

    def test_accepts_roundoff_level_asymmetry_and_stores_symmetric_covariance(self):
        covariance = np.array([[1.0, 0.5 + 5e-11], [0.5, 1.0]])

        hypothesis = WeightedGaussianHypothesis(
            mean=np.array([0.0, 0.0]),
            covariance=covariance,
        )

        expected = 0.5 * (covariance + covariance.T)
        np.testing.assert_array_equal(hypothesis.covariance, expected)
        np.testing.assert_array_equal(
            hypothesis.covariance,
            hypothesis.covariance.T,
        )

    def test_accepts_positive_semidefinite_covariance(self):
        hypothesis = WeightedGaussianHypothesis(
            mean=np.array([0.0, 0.0]),
            covariance=np.array([[1.0, 1.0], [1.0, 1.0]]),
        )

        self.assertTrue(np.allclose(hypothesis.covariance, np.ones((2, 2))))


if __name__ == "__main__":
    unittest.main()
