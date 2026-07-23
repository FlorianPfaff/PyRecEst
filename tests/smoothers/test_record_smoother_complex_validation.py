"""Regression tests for real-valued record smoother inputs."""

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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "state",
            np.array([1.0 + 2.0j, 0.0]),
            "record states must contain real values",
        ),
        (
            "state",
            np.array([1.0 + 2.0j, 0.0], dtype=object),
            "record states must contain real values",
        ),
        (
            "covariance",
            np.array([[1.0 + 2.0j, 0.0], [0.0, 1.0]]),
            "record covariances must contain real values",
        ),
        (
            "covariance",
            np.array([[1.0 + 2.0j, 0.0], [0.0, 1.0]], dtype=object),
            "record covariances must contain real values",
        ),
    ],
)
def test_rejects_complex_record_values(field, value, message) -> None:
    records = _records()
    records[0][field] = value

    with pytest.raises(ValueError, match=message):
        smooth_records(
            records,
            method="rts",
            transition_model=_transition,
            process_noise_model=_process_noise,
        )


@pytest.mark.parametrize("dtype", [complex, object])
@pytest.mark.parametrize("model_name", ["transition_model", "process_noise_model"])
def test_rejects_complex_model_matrices(dtype, model_name) -> None:
    def complex_matrix(_dt: float, _state_dim: int) -> np.ndarray:
        return np.array([[1.0 + 2.0j, 0.0], [0.0, 1.0]], dtype=dtype)

    models = {
        "transition_model": _transition,
        "process_noise_model": _process_noise,
    }
    models[model_name] = complex_matrix

    with pytest.raises(ValueError, match=rf"{model_name} must return real values"):
        smooth_records(_records(), method="rts", **models)
