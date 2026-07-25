import numpy as np
from pyrecest.filters import (
    WeightedGaussianHypothesis,
    moment_match_gaussian_hypotheses,
)


def test_maximum_finite_covariance_survives_symmetrization_and_moment_matching():
    maximum = np.finfo(float).max
    hypothesis = WeightedGaussianHypothesis(
        mean=np.array([0.0]),
        covariance=np.array([[maximum]]),
    )

    mean, covariance, weights = moment_match_gaussian_hypotheses([hypothesis])

    np.testing.assert_array_equal(hypothesis.covariance, np.array([[maximum]]))
    np.testing.assert_array_equal(mean, np.array([0.0]))
    np.testing.assert_array_equal(covariance, np.array([[maximum]]))
    np.testing.assert_array_equal(weights, np.array([1.0]))
