from __future__ import annotations

import numpy as np
import pytest
from pyrecest.smoothers import smooth_records


def _identity_transition(_dt: float, state_dim: int) -> np.ndarray:
    return np.eye(state_dim)


def _zero_process_noise(_dt: float, state_dim: int) -> np.ndarray:
    return np.zeros((state_dim, state_dim))


def _records(covariance: np.ndarray) -> list[dict[str, object]]:
    return [
        {
            "time_s": 0.0,
            "state": np.array([0.0, 0.0]),
            "covariance": covariance.copy(),
        },
        {
            "time_s": 1.0,
            "state": np.array([0.0, 0.0]),
            "covariance": covariance.copy(),
        },
    ]


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        (np.array([[1.0, 3.0], [-3.0, 1.0]]), "symmetric"),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), "positive semidefinite"),
    ],
)
def test_record_smoother_rejects_invalid_covariance_structure(
    covariance: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        smooth_records(
            _records(covariance),
            method="rts",
            transition_model=_identity_transition,
            process_noise_model=_zero_process_noise,
        )


def test_record_smoother_accepts_roundoff_level_covariance_skew() -> None:
    covariance = np.array([[1.0, 1.0e-12], [-1.0e-12, 1.0]])

    result = smooth_records(
        _records(covariance),
        method="rts",
        transition_model=_identity_transition,
        process_noise_model=_zero_process_noise,
    )

    assert np.allclose(result[0]["covariance"], np.eye(2))
