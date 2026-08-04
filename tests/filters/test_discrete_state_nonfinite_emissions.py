"""Regression tests for invalid non-finite discrete-state emissions."""

import numpy as np
import pytest

from pyrecest.filters.discrete_state import (
    discrete_forward_backward,
    scaled_emissions,
)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
def test_scaled_emissions_rejects_invalid_nonfinite_values(invalid_value):
    log_likelihood = np.array([[invalid_value, 0.0]])

    with pytest.raises(
        ValueError,
        match=r"log_likelihood must contain finite values or -np\.inf",
    ):
        scaled_emissions(log_likelihood)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
def test_forward_backward_rejects_invalid_nonfinite_values(invalid_value):
    log_likelihood = np.array([[invalid_value, 0.0]])

    with pytest.raises(
        ValueError,
        match=r"log_likelihood must contain finite values or -np\.inf",
    ):
        discrete_forward_backward(log_likelihood, np.eye(2))


def test_scaled_emissions_still_preserves_negative_infinity_as_zero_mass():
    scaled, offsets = scaled_emissions(np.array([[-np.inf, 0.0]]))

    np.testing.assert_array_equal(scaled, np.array([[0.0, 1.0]]))
    np.testing.assert_array_equal(offsets, np.array([0.0]))
