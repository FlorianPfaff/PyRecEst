"""Regression tests for exact MEM-QKF fixed-lag validation."""

import numpy as np
import pytest

from pyrecest.backend import array, eye
from pyrecest.smoothers import FixedLagMEMQKFSmoother, MEMQKFTrackerState


def _single_state() -> MEMQKFTrackerState:
    return MEMQKFTrackerState(
        kinematic_state=array([0.0]),
        covariance=array([[1.0]]),
        shape_state=array([0.0, 2.0, 1.0]),
        shape_covariance=eye(3),
    )


def test_constructor_rejects_fractional_or_boolean_lag():
    for lag in (0.5, True):
        with pytest.raises(ValueError, match="lag must be a non-negative integer"):
            FixedLagMEMQKFSmoother(lag=lag)


def test_smooth_rejects_fractional_lag_override():
    smoother = FixedLagMEMQKFSmoother(lag=0)

    with pytest.raises(ValueError, match="lag must be a non-negative integer"):
        smoother.smooth([_single_state()], lag=0.5)


def test_numpy_integer_lags_remain_supported():
    smoother = FixedLagMEMQKFSmoother(lag=np.int64(1))
    smoothed, gains = smoother.smooth([_single_state()], lag=np.int64(0))

    assert smoother.lag == 1
    assert len(smoothed) == 1
    assert gains == [[]]
