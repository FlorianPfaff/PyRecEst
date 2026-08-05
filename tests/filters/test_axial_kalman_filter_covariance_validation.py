"""Regression tests for axial Kalman covariance validation."""

import numpy.testing as npt
import pytest

from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.axial_kalman_filter import AxialKalmanFilter


def _unchecked_gaussian(covariance):
    return GaussianDistribution(
        array([1.0, 0.0]),
        array(covariance),
        check_validity=False,
    )


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        ([[1.0, 0.5], [0.0, 1.0]], "symmetric"),
        ([[1.0, 0.0], [0.0, -0.25]], "positive semidefinite"),
    ],
)
def test_filter_state_rejects_invalid_covariance_without_mutation(
    covariance,
    message,
):
    axial_filter = AxialKalmanFilter()
    original_mu = axial_filter.filter_state.mu.copy()
    original_covariance = axial_filter.filter_state.C.copy()

    with pytest.raises(ValueError, match=message):
        axial_filter.filter_state = _unchecked_gaussian(covariance)

    npt.assert_array_equal(axial_filter.filter_state.mu, original_mu)
    npt.assert_array_equal(axial_filter.filter_state.C, original_covariance)


def test_prediction_rejects_invalid_noise_covariance_before_broadcasting():
    axial_filter = AxialKalmanFilter()
    axial_filter.filter_state = _unchecked_gaussian([[0.5, 0.0], [0.0, 0.5]])
    original_mu = axial_filter.filter_state.mu.copy()
    original_covariance = axial_filter.filter_state.C.copy()

    with pytest.raises(ValueError, match="system noise covariance.*symmetric"):
        axial_filter.predict_identity(
            _unchecked_gaussian([[0.25, 0.1], [0.0, 0.25]])
        )

    npt.assert_array_equal(axial_filter.filter_state.mu, original_mu)
    npt.assert_array_equal(axial_filter.filter_state.C, original_covariance)


def test_filter_state_accepts_singular_positive_semidefinite_covariance():
    axial_filter = AxialKalmanFilter()
    covariance = array([[0.5, 0.0], [0.0, 0.0]])

    axial_filter.filter_state = GaussianDistribution(
        array([1.0, 0.0]),
        covariance,
        check_validity=False,
    )

    npt.assert_array_equal(axial_filter.filter_state.C, covariance)
