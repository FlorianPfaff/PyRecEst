from __future__ import annotations

import numpy as np
import pytest
from pyrecest.smoothers.record_smoother import fixed_lag_smooth_records, smooth_records


def _transition(dt: float, state_dim: int) -> np.ndarray:
    assert state_dim == 2
    return np.array([[1.0, dt], [0.0, 1.0]])


def _process_noise(dt: float, state_dim: int) -> np.ndarray:
    assert state_dim == 2
    return 0.01 * max(dt, 0.0) * np.eye(2)


def _records() -> list[dict[str, object]]:
    return [
        {
            "time_s": 0.0,
            "state": np.array([0.0, 1.0]),
            "covariance": np.diag([2.0, 1.0]),
            "source": "radar",
            "action": "updated",
        },
        {
            "time_s": 1.0,
            "state": np.array([1.3, 1.0]),
            "covariance": np.diag([1.0, 0.8]),
            "source": "rf",
            "action": "updated",
        },
        {
            "time_s": 2.0,
            "state": np.array([2.0, 1.0]),
            "covariance": np.diag([0.5, 0.4]),
            "source": "radar",
            "action": "updated",
        },
    ]


def test_fixed_lag_smooth_records_preserves_metadata_and_filtered_state() -> None:
    out = fixed_lag_smooth_records(
        _records(),
        transition_model=_transition,
        process_noise_model=_process_noise,
        lag=2.0,
        metadata={"smoother_method": "fixed-lag", "smoother_lag_s": 2.0},
    )

    assert len(out) == 3
    assert out[0]["source"] == "radar"
    assert out[1]["action"] == "updated"
    assert out[0]["smoother_method"] == "fixed-lag"
    assert out[0]["smoother_lag_s"] == 2.0
    assert np.allclose(out[0]["filtered_state"], np.array([0.0, 1.0]))
    assert out[0]["state"].shape == (2,)
    assert out[0]["covariance"].shape == (2, 2)
    assert not np.allclose(out[0]["state"], out[0]["filtered_state"])


def test_rts_and_fixed_lag_match_when_lag_covers_all_future_records() -> None:
    full = smooth_records(
        _records(),
        method="rts",
        transition_model=_transition,
        process_noise_model=_process_noise,
    )
    lagged = smooth_records(
        _records(),
        method="fixed-lag",
        transition_model=_transition,
        process_noise_model=_process_noise,
        lag=100.0,
    )

    assert np.allclose(full[0]["state"], lagged[0]["state"])
    assert np.allclose(full[0]["covariance"], lagged[0]["covariance"])


def test_none_returns_copied_records_with_metadata() -> None:
    records = _records()
    out = smooth_records(
        records,
        method="none",
        transition_model=_transition,
        process_noise_model=_process_noise,
        metadata={"smoother_method": "none"},
    )

    assert out[0]["source"] == records[0]["source"]
    assert out[0]["smoother_method"] == "none"
    assert out is not records
    assert out[0] is not records[0]
    out[0]["state"][0] = 99.0
    assert records[0]["state"][0] == 0.0


def test_fixed_lag_requires_nonnegative_lag() -> None:
    with pytest.raises(ValueError, match="nonnegative lag"):
        smooth_records(
            _records(),
            method="fixed-lag",
            transition_model=_transition,
            process_noise_model=_process_noise,
            lag=None,
        )


def test_records_must_be_sorted() -> None:
    records = list(reversed(_records()))
    with pytest.raises(ValueError, match="sorted"):
        smooth_records(
            records,
            method="rts",
            transition_model=_transition,
            process_noise_model=_process_noise,
        )
