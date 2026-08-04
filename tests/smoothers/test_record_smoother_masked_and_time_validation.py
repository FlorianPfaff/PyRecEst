"""Regression tests for masked and complex record smoother controls."""

from __future__ import annotations

import numpy as np
import pytest
from pyrecest.smoothers.record_smoother import smooth_records


def _records() -> list[dict[str, object]]:
    return [
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
    ]


def _transition(dt: float, _state_dim: int) -> np.ndarray:
    return np.array([[1.0, dt], [0.0, 1.0]])


def _process_noise(_dt: float, state_dim: int) -> np.ndarray:
    return np.zeros((state_dim, state_dim))


def _smooth(records, **models):
    defaults = {
        "transition_model": _transition,
        "process_noise_model": _process_noise,
    }
    defaults.update(models)
    return smooth_records(records, method="rts", **defaults)


def test_rejects_numpy_complex_record_time() -> None:
    records = _records()
    records[1]["time_s"] = np.complex64(1.0 + 2.0j)

    with pytest.raises(ValueError, match="record times must contain real values"):
        _smooth(records)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "time_s",
            np.ma.array(0.0, mask=True),
            "record times must not contain masked values",
        ),
        (
            "state",
            np.ma.array([0.0, 1.0], mask=[True, False]),
            "record states must not contain masked values",
        ),
        (
            "state",
            np.array([np.ma.masked, 1.0], dtype=object),
            "record states must not contain masked values",
        ),
        (
            "covariance",
            np.ma.array(np.eye(2), mask=[[True, False], [False, False]]),
            "record covariances must not contain masked values",
        ),
    ],
)
def test_rejects_masked_record_values(field, value, message) -> None:
    records = _records()
    records[0][field] = value

    with pytest.raises(ValueError, match=message):
        _smooth(records)


@pytest.mark.parametrize("model_name", ["transition_model", "process_noise_model"])
def test_rejects_masked_model_matrices(model_name) -> None:
    def masked_matrix(_dt: float, _state_dim: int) -> np.ndarray:
        return np.ma.array(
            np.eye(2),
            mask=[[True, False], [False, False]],
        )

    with pytest.raises(
        ValueError,
        match=rf"{model_name} must not return masked values",
    ):
        _smooth(_records(), **{model_name: masked_matrix})


def test_accepts_clear_mask_wrappers() -> None:
    records = _records()
    records[0]["time_s"] = np.ma.array(0.0, mask=False)
    records[0]["state"] = np.ma.array([0.0, 1.0], mask=False)
    records[0]["covariance"] = np.ma.array(np.eye(2), mask=False)

    def clear_mask_transition(dt: float, _state_dim: int) -> np.ndarray:
        return np.ma.array([[1.0, dt], [0.0, 1.0]], mask=False)

    smoothed = _smooth(records, transition_model=clear_mask_transition)

    assert len(smoothed) == len(records)
    assert np.isfinite(smoothed[0]["state"]).all()
