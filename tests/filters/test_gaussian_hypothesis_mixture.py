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
