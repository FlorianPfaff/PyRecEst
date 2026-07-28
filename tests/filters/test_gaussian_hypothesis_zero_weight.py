import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.filters import (
    WeightedGaussianHypothesis,
    moment_match_gaussian_hypotheses,
)


class GaussianHypothesisZeroWeightTest(unittest.TestCase):
    def test_zero_weight_hypothesis_cannot_contaminate_covariance(self):
        hypotheses = [
            WeightedGaussianHypothesis(
                mean=np.array([0.0]),
                covariance=np.array([[1.0]]),
                log_weight=0.0,
            ),
            WeightedGaussianHypothesis(
                mean=np.array([1e308]),
                covariance=np.array([[1.0]]),
                log_weight=-np.inf,
            ),
        ]

        with np.errstate(over="raise", invalid="raise"):
            mean, covariance, weights = moment_match_gaussian_hypotheses(hypotheses)

        npt.assert_array_equal(weights, np.array([1.0, 0.0]))
        npt.assert_array_equal(mean, np.array([0.0]))
        npt.assert_array_equal(covariance, np.array([[1.0]]))


if __name__ == "__main__":
    unittest.main()
