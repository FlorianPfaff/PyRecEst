# pylint: disable=no-name-in-module,no-member
"""Tests for robust linear-Gaussian helper functions."""

import pytest

from pyrecest.backend import allclose, array
from pyrecest.filters._linear_gaussian import huber_covariance_scale


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
