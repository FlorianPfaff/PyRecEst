from __future__ import annotations

import numpy as np
import pytest
from pyrecest.smoothers.record_smoother import fixed_lag_smooth_records


def _identity_transition(_dt: float, state_dim: int) -> np.ndarray:
    return np.eye(state_dim)


def _zero_process_noise(_dt: float, state_dim: int) -> np.ndarray:
    return np.zeros((state_dim, state_dim))


def test_fixed_lag_rejects_subtly_decreasing_timestamps() -> None:
    records = [
        {
            "time_s": 0.0,
            "state": np.array([0.0, 1.0]),
            "covariance": np.eye(2),
        },
        {
            "time_s": 1.0,
            "state": np.array([1.0, 1.0]),
            "covariance": np.eye(2),
        },
        {
            "time_s": np.nextafter(1.0, 0.0),
            "state": np.array([1.0, 1.0]),
            "covariance": np.eye(2),
        },
    ]

    with pytest.raises(ValueError, match="sorted by nondecreasing time"):
        fixed_lag_smooth_records(
            records,
            transition_model=_identity_transition,
            process_noise_model=_zero_process_noise,
            lag=0.0,
        )
