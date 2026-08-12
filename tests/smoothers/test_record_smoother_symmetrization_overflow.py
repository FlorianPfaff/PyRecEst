from __future__ import annotations

import numpy as np
from pyrecest.smoothers import smooth_records


def _identity_transition(_dt: float, state_dim: int) -> np.ndarray:
    return np.eye(state_dim)


def _large_process_noise(_dt: float, state_dim: int) -> np.ndarray:
    magnitude = 0.75 * np.finfo(float).max
    return np.eye(state_dim) * magnitude


def test_record_smoother_symmetrizes_large_finite_process_noise_without_overflow() -> (
    None
):
    records = [
        {
            "time_s": 0.0,
            "state": np.array([0.0]),
            "covariance": np.zeros((1, 1)),
        },
        {
            "time_s": 1.0,
            "state": np.array([0.0]),
            "covariance": np.zeros((1, 1)),
        },
    ]

    with np.errstate(over="raise", invalid="raise"):
        result = smooth_records(
            records,
            method="rts",
            transition_model=_identity_transition,
            process_noise_model=_large_process_noise,
        )

    assert np.isfinite(result[0]["covariance"]).all()
    assert np.array_equal(result[0]["covariance"], np.zeros((1, 1)))
