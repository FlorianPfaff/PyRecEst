import numpy as np
from pyrecest.filters import (
    WeightedGaussianHypothesis,
    moment_match_gaussian_hypotheses,
)


def test_weighted_gaussian_hypothesis_copies_mean_input():
    source_mean = np.array([1.0, 2.0])
    hypothesis = WeightedGaussianHypothesis(source_mean, np.eye(2))

    source_mean[:] = np.array([10.0, 20.0])

    np.testing.assert_array_equal(hypothesis.mean, np.array([1.0, 2.0]))
    matched_mean, _, _ = moment_match_gaussian_hypotheses([hypothesis])
    np.testing.assert_array_equal(matched_mean, np.array([1.0, 2.0]))
