"""Record-compatible RTS and fixed-lag smoothing utilities.

The helpers in this module smooth timestamped Kalman posterior records without
assuming a particular tracker class.  They are intended for asynchronous
multi-sensor replay pipelines where each record carries source/action metadata
that should be preserved while the state and covariance are smoothed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

SmootherMethod = Literal["none", "rts", "fixed-lag"]
TransitionModel = Callable[..., np.ndarray]
ProcessNoiseModel = Callable[..., np.ndarray]


@dataclass(frozen=True)
class RecordSmootherConfig:
    """Configuration for :func:`smooth_records`.

    Parameters
    ----------
    method:
        ``"rts"`` runs a full Rauch--Tung--Striebel backward pass.  ``"fixed-lag"``
        applies the same backward recursion only over future records within
        ``lag`` seconds.  ``"none"`` returns copied records.
    lag:
        Nonnegative fixed-lag horizon in seconds. Required for ``"fixed-lag"``.
    time_key, state_key, covariance_key:
        Keys used to extract timestamp, filtered state, and filtered covariance.
    output_state_key, output_covariance_key:
        Keys used for the smoothed posterior in returned records.
    filtered_state_key, filtered_covariance_key:
        Keys used to preserve the original filtered posterior in returned records.
    metadata:
        Extra key/value pairs appended to every returned record.
    """

    method: SmootherMethod = "fixed-lag"
    lag: float | None = None
    time_key: str = "time_s"
    state_key: str = "state"
    covariance_key: str = "covariance"
    output_state_key: str = "state"
    output_covariance_key: str = "covariance"
    filtered_state_key: str = "filtered_state"
    filtered_covariance_key: str = "filtered_covariance"
    metadata: Mapping[str, Any] | None = None


__all__ = [
    "RecordSmootherConfig",
    "SmootherMethod",
    "fixed_lag_smooth_records",
    "rts_smooth_records",
    "smooth_records",
]


def smooth_records(
    records: Sequence[Mapping[str, Any]],
    *,
    method: SmootherMethod = "fixed-lag",
    transition_model: TransitionModel,
    process_noise_model: ProcessNoiseModel,
    lag: float | None = None,
    time_key: str = "time_s",
    state_key: str = "state",
    covariance_key: str = "covariance",
    output_state_key: str = "state",
    output_covariance_key: str = "covariance",
    filtered_state_key: str = "filtered_state",
    filtered_covariance_key: str = "filtered_covariance",
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return records with smoothed state/covariance estimates.

    ``records`` may contain arbitrary extra metadata such as sensor source,
    accepted/rejected action, NIS, residuals, association IDs, or measurement
    dimensions.  Returned records preserve that metadata and additionally store
    the original filtered state/covariance under ``filtered_*`` keys.

    ``transition_model`` and ``process_noise_model`` are callables that accept
    ``dt`` seconds.  They may optionally accept a second ``state_dim`` argument;
    this allows one function to support 6D CV, 7D bias states, or other linear
    state layouts.
    """

    config = RecordSmootherConfig(
        method=method,
        lag=lag,
        time_key=time_key,
        state_key=state_key,
        covariance_key=covariance_key,
        output_state_key=output_state_key,
        output_covariance_key=output_covariance_key,
        filtered_state_key=filtered_state_key,
        filtered_covariance_key=filtered_covariance_key,
        metadata=metadata,
    )
    return _smooth_records_with_config(
        records,
        transition_model=transition_model,
        process_noise_model=process_noise_model,
        config=config,
    )


def rts_smooth_records(
    records: Sequence[Mapping[str, Any]],
    *,
    transition_model: TransitionModel,
    process_noise_model: ProcessNoiseModel,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run a full Rauch--Tung--Striebel pass over timestamped records."""

    return smooth_records(
        records,
        method="rts",
        transition_model=transition_model,
        process_noise_model=process_noise_model,
        **kwargs,
    )


def fixed_lag_smooth_records(
    records: Sequence[Mapping[str, Any]],
    *,
    transition_model: TransitionModel,
    process_noise_model: ProcessNoiseModel,
    lag: float,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run fixed-lag RTS smoothing over timestamped records."""

    return smooth_records(
        records,
        method="fixed-lag",
        transition_model=transition_model,
        process_noise_model=process_noise_model,
        lag=lag,
        **kwargs,
    )


def _smooth_records_with_config(
    records: Sequence[Mapping[str, Any]],
    *,
    transition_model: TransitionModel,
    process_noise_model: ProcessNoiseModel,
    config: RecordSmootherConfig,
) -> list[dict[str, Any]]:
    if config.method not in ("none", "rts", "fixed-lag"):
        raise ValueError(f"unknown smoothing method {config.method!r}")
    if config.method == "fixed-lag" and (config.lag is None or config.lag < 0.0):
        raise ValueError("fixed-lag smoothing requires a nonnegative lag")
    if not records:
        return []

    copied = [_copy_record(record) for record in records]
    metadata = {} if config.metadata is None else dict(config.metadata)
    if config.method == "none":
        for item in copied:
            item.update(metadata)
        return copied

    times, filtered_states, filtered_covariances = _record_arrays(
        copied,
        time_key=config.time_key,
        state_key=config.state_key,
        covariance_key=config.covariance_key,
    )
    if config.method == "rts":
        smoothed_states, smoothed_covariances = _rts_smooth_arrays(
            times,
            filtered_states,
            filtered_covariances,
            transition_model=transition_model,
            process_noise_model=process_noise_model,
            start_index=0,
            end_index=len(copied) - 1,
        )
    else:
        smoothed_states, smoothed_covariances = _fixed_lag_smooth_arrays(
            times,
            filtered_states,
            filtered_covariances,
            transition_model=transition_model,
            process_noise_model=process_noise_model,
            lag=float(config.lag),
        )

    out: list[dict[str, Any]] = []
    for idx, record in enumerate(copied):
        item = _copy_record(record)
        item[config.filtered_state_key] = filtered_states[idx].copy()
        item[config.filtered_covariance_key] = filtered_covariances[idx].copy()
        item[config.output_state_key] = smoothed_states[idx].copy()
        item[config.output_covariance_key] = smoothed_covariances[idx].copy()
        item.update(metadata)
        out.append(item)
    return out


def _record_arrays(
    records: Sequence[Mapping[str, Any]],
    *,
    time_key: str,
    state_key: str,
    covariance_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray([float(record[time_key]) for record in records], dtype=float)
    if not np.isfinite(times).all():
        raise ValueError("record times must be finite")
    if np.any(np.diff(times) < -1.0e-9):
        raise ValueError("records must be sorted by nondecreasing time")

    states = [
        np.asarray(record[state_key], dtype=float).reshape(-1) for record in records
    ]
    state_dim = states[0].size
    if state_dim == 0:
        raise ValueError("state vectors must be nonempty")
    if any(state.size != state_dim for state in states):
        raise ValueError("all records must have the same state dimension")
    state_array = np.stack(states)
    if not np.isfinite(state_array).all():
        raise ValueError("record states must be finite")

    covariances = []
    for record in records:
        covariance = np.asarray(record[covariance_key], dtype=float)
        if covariance.shape != (state_dim, state_dim):
            raise ValueError("record covariance shape must match state dimension")
        if not np.isfinite(covariance).all():
            raise ValueError("record covariances must be finite")
        covariances.append(_symmetrized(covariance))
    return times, state_array, np.stack(covariances)


def _fixed_lag_smooth_arrays(
    times: np.ndarray,
    filtered_states: np.ndarray,
    filtered_covariances: np.ndarray,
    *,
    transition_model: TransitionModel,
    process_noise_model: ProcessNoiseModel,
    lag: float,
) -> tuple[np.ndarray, np.ndarray]:
    smoothed_states = filtered_states.copy()
    smoothed_covariances = filtered_covariances.copy()
    for start_index, time_s in enumerate(times):
        end_index = int(np.searchsorted(times, time_s + lag, side="right") - 1)
        if end_index <= start_index:
            continue
        states, covariances = _rts_smooth_arrays(
            times,
            filtered_states,
            filtered_covariances,
            transition_model=transition_model,
            process_noise_model=process_noise_model,
            start_index=start_index,
            end_index=end_index,
        )
        smoothed_states[start_index] = states[start_index]
        smoothed_covariances[start_index] = covariances[start_index]
    return smoothed_states, smoothed_covariances


def _rts_smooth_arrays(
    times: np.ndarray,
    filtered_states: np.ndarray,
    filtered_covariances: np.ndarray,
    *,
    transition_model: TransitionModel,
    process_noise_model: ProcessNoiseModel,
    start_index: int,
    end_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= start_index <= end_index < len(times):
        raise ValueError("invalid smoothing interval")
    smoothed_states = filtered_states.copy()
    smoothed_covariances = filtered_covariances.copy()
    state_dim = int(filtered_states.shape[1])
    for idx in range(end_index - 1, start_index - 1, -1):
        transition, predicted_state, predicted_covariance = _predict_arrays(
            times,
            filtered_states,
            filtered_covariances,
            idx,
            state_dim=state_dim,
            transition_model=transition_model,
            process_noise_model=process_noise_model,
        )
        gain = _smoothing_gain(
            filtered_covariances[idx], transition, predicted_covariance
        )
        smoothed_states[idx] = filtered_states[idx] + gain @ (
            smoothed_states[idx + 1] - predicted_state
        )
        smoothed_covariances[idx] = _symmetrized(
            filtered_covariances[idx]
            + gain @ (smoothed_covariances[idx + 1] - predicted_covariance) @ gain.T
        )
    return smoothed_states, smoothed_covariances


def _predict_arrays(
    times: np.ndarray,
    filtered_states: np.ndarray,
    filtered_covariances: np.ndarray,
    index: int,
    *,
    state_dim: int,
    transition_model: TransitionModel,
    process_noise_model: ProcessNoiseModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = float(times[index + 1] - times[index])
    if dt < -1.0e-9:
        raise ValueError("smoothing records must be sorted by time")
    dt = max(0.0, dt)
    transition = _call_model(transition_model, dt, state_dim, "transition_model")
    process_noise = _call_model(
        process_noise_model, dt, state_dim, "process_noise_model"
    )
    predicted_state = transition @ filtered_states[index]
    predicted_covariance = _symmetrized(
        transition @ filtered_covariances[index] @ transition.T + process_noise
    )
    return transition, predicted_state, predicted_covariance


def _call_model(
    model: Callable[..., np.ndarray], dt: float, state_dim: int, name: str
) -> np.ndarray:
    try:
        matrix = model(dt, state_dim)
    except TypeError as exc:
        try:
            matrix = model(dt)
        except TypeError:
            raise exc
    array = np.asarray(matrix, dtype=float)
    if array.shape != (state_dim, state_dim):
        raise ValueError(f"{name} must return a ({state_dim}, {state_dim}) matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must return finite values")
    return array


def _smoothing_gain(
    filtered_covariance: np.ndarray,
    transition: np.ndarray,
    predicted_covariance: np.ndarray,
) -> np.ndarray:
    right = filtered_covariance @ transition.T
    try:
        return np.linalg.solve(predicted_covariance.T, right.T).T
    except np.linalg.LinAlgError:
        return right @ np.linalg.pinv(predicted_covariance)


def _copy_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in record.items()
    }


def _symmetrized(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)
