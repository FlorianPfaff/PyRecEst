import unittest

import numpy as np
from pyrecest.filters import (
    WeightedGaussianHypothesis,
    moment_match_gaussian_hypotheses,
    normalize_log_weights,
)


class GaussianHypothesisMixtureTest(unittest.TestCase):
    def test_normalize_log_weights_is_stable(self):
        weights = normalize_log_weights(np.array([1000.0, 1000.0]))

        self.assertTrue(np.allclose(weights, np.array([0.5, 0.5])))

    def test_positive_infinite_log_weights_dominate(self):
        weights = normalize_log_weights(np.array([0.0, np.inf, -np.inf]))

        self.assertTrue(np.allclose(weights, np.array([0.0, 1.0, 0.0])))

    def test_multiple_positive_infinite_log_weights_share_mass(self):
        weights = normalize_log_weights(np.array([np.inf, 5.0, np.inf]))

        self.assertTrue(np.allclose(weights, np.array([0.5, 0.0, 0.5])))

    def test_nan_log_weights_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "NaN"):
            normalize_log_weights(np.array([0.0, np.nan]))

        with self.assertRaisesRegex(ValueError, "NaN"):
            moment_match_gaussian_hypotheses(
                [
                    WeightedGaussianHypothesis(
                        np.array([0.0]), np.array([[1.0]]), log_weight=0.0
                    ),
                    WeightedGaussianHypothesis(
                        np.array([1.0]), np.array([[1.0]]), log_weight=np.nan
                    ),
                ]
            )

    def test_hypotheses_reject_nonfinite_mean_and_covariance(self):
        with self.assertRaisesRegex(ValueError, "mean"):
            WeightedGaussianHypothesis(np.array([np.nan]), np.array([[1.0]]))

        with self.assertRaisesRegex(ValueError, "mean"):
            WeightedGaussianHypothesis(np.array([np.inf]), np.array([[1.0]]))

        with self.assertRaisesRegex(ValueError, "covariance"):
            WeightedGaussianHypothesis(np.array([0.0]), np.array([[np.nan]]))

        with self.assertRaisesRegex(ValueError, "covariance"):
            WeightedGaussianHypothesis(np.array([0.0]), np.array([[np.inf]]))

    def test_moment_matching_rejects_mixed_state_dimensions(self):
        with self.assertRaisesRegex(ValueError, "same dimension"):
            moment_match_gaussian_hypotheses(
                [
                    WeightedGaussianHypothesis(np.array([0.0]), np.array([[1.0]])),
                    WeightedGaussianHypothesis(
                        np.array([1.0, 2.0]),
                        np.eye(2),
                    ),
                ]
            )

    def test_moment_matching_respects_dominant_infinite_weight(self):
        mean, covariance, weights = moment_match_gaussian_hypotheses(
            [
                WeightedGaussianHypothesis(
                    np.array([0.0]), np.array([[1.0]]), log_weight=0.0
                ),
                WeightedGaussianHypothesis(
                    np.array([3.0]), np.array([[2.0]]), log_weight=np.inf
                ),
            ]
        )

        self.assertTrue(np.allclose(weights, np.array([0.0, 1.0])))
        self.assertTrue(np.allclose(mean, np.array([3.0])))
        self.assertTrue(np.allclose(covariance, np.array([[2.0]])))

    def test_moment_matching_includes_between_hypothesis_spread(self):
        mean, covariance, weights = moment_match_gaussian_hypotheses(
            [
                WeightedGaussianHypothesis(np.array([0.0]), np.array([[1.0]])),
                WeightedGaussianHypothesis(np.array([2.0]), np.array([[1.0]])),
            ]
        )

        self.assertTrue(np.allclose(weights, np.array([0.5, 0.5])))
        self.assertTrue(np.allclose(mean, np.array([1.0])))
        self.assertTrue(np.allclose(covariance, np.array([[2.0]])))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            normalize_log_weights([])

        with self.assertRaises(ValueError):
            WeightedGaussianHypothesis(np.array([0.0, 1.0]), np.array([[1.0]]))

        with self.assertRaises(ValueError):
            moment_match_gaussian_hypotheses([])


if __name__ == "__main__":
    unittest.main()
