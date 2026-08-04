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


def test_tiny_positive_weight_keeps_extreme_mean_variance_finite():
    maximum = np.finfo(float).max
    mean_magnitude = 0.75 * maximum
    tiny_weight = 1.0e-310
    hypotheses = [
        WeightedGaussianHypothesis(
            mean=np.array([mean_magnitude]),
            covariance=np.zeros((1, 1)),
            log_weight=np.log(tiny_weight),
        ),
        WeightedGaussianHypothesis(
            mean=np.array([-mean_magnitude]),
            covariance=np.zeros((1, 1)),
        ),
    ]

    with np.errstate(over="raise", invalid="raise"):
        mean, covariance, weights = moment_match_gaussian_hypotheses(hypotheses)

    expected_variance = (2.0 * np.sqrt(weights[0]) * mean_magnitude) ** 2
    assert weights[0] > 0.0
    assert np.all(np.isfinite(covariance))
    np.testing.assert_array_equal(mean, np.array([-mean_magnitude]))
    np.testing.assert_allclose(
        covariance,
        np.array([[expected_variance]]),
        rtol=1.0e-12,
    )
