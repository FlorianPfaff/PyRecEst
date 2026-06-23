# pylint: disable=no-name-in-module,no-member
"""Tests for robust linear-Gaussian helper functions."""

import pytest
from pyrecest.backend import allclose, array
from pyrecest.filters._linear_gaussian import (
    huber_covariance_scale,
    linear_gaussian_update,
    student_t_covariance_scale,
)


def test_huber_covariance_scale_leaves_inliers_unchanged():
    assert allclose(huber_covariance_scale(1.0, huber_threshold=2.0), 1.0)


def test_huber_covariance_scale_downweights_outliers():
    assert allclose(huber_covariance_scale(9.0, huber_threshold=2.0), 1.5)


def test_huber_covariance_scale_vectorizes_over_nis():
    scale = huber_covariance_scale(array([1.0, 4.0, 16.0]), huber_threshold=2.0)

    assert allclose(scale, array([1.0, 1.0, 2.0]))


def test_huber_covariance_scale_rejects_nonpositive_threshold():
    with pytest.raises(ValueError, match="positive"):
        huber_covariance_scale(1.0, huber_threshold=0.0)


@pytest.mark.parametrize(
    "invalid_threshold", [float("nan"), float("inf"), -float("inf")]
)
def test_huber_covariance_scale_rejects_nonfinite_threshold(invalid_threshold):
    with pytest.raises(ValueError, match="finite and positive"):
        huber_covariance_scale(1.0, huber_threshold=invalid_threshold)


@pytest.mark.parametrize("invalid_dof", [float("nan"), float("inf"), -float("inf")])
def test_student_t_covariance_scale_rejects_nonfinite_dof(invalid_dof):
    with pytest.raises(ValueError, match="finite and greater than 2"):
        student_t_covariance_scale(1.0, measurement_dim=1, dof=invalid_dof)


@pytest.mark.parametrize(
    "invalid_min_scale",
    [float("nan"), float("inf"), -float("inf")],
)
def test_student_t_covariance_scale_rejects_nonfinite_min_scale(invalid_min_scale):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        student_t_covariance_scale(
            1.0,
            measurement_dim=1,
            min_scale=invalid_min_scale,
        )


@pytest.mark.parametrize("invalid_scale", [float("nan"), float("inf"), -float("inf")])
def test_linear_gaussian_update_rejects_nonfinite_scale(invalid_scale):
    with pytest.raises(ValueError, match="scale must be finite and positive"):
        linear_gaussian_update(
            array([0.0]),
            array([[1.0]]),
            array([0.0]),
            array([[1.0]]),
            array([[1.0]]),
            scale=invalid_scale,
        )
