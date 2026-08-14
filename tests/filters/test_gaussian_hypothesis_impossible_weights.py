"""Regression tests for impossible Gaussian-hypothesis weights."""

import numpy as np
import pytest
from pyrecest.filters import (
    WeightedGaussianHypothesis,
    moment_match_gaussian_hypotheses,
    normalize_log_weights,
)


def test_all_negative_infinite_log_weights_are_rejected():
    with pytest.raises(ValueError, match="positive total mass"):
        normalize_log_weights(np.array([-np.inf, -np.inf]))


def test_moment_matching_rejects_hypotheses_with_zero_total_mass():
    hypotheses = [
        WeightedGaussianHypothesis(
            np.array([0.0]),
            np.array([[1.0]]),
            log_weight=-np.inf,
        ),
        WeightedGaussianHypothesis(
            np.array([10.0]),
            np.array([[1.0]]),
            log_weight=-np.inf,
        ),
    ]

    with pytest.raises(ValueError, match="positive total mass"):
        moment_match_gaussian_hypotheses(hypotheses)
