"""Calibration helpers for asynchronous sensor-fusion workflows."""

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from . import time_offset as _time_offset_module

_REJECTED_REAL_SCALAR_TYPES = (
    bool,
    np.bool_,
    str,
    bytes,
    bytearray,
    complex,
    np.complexfloating,
)


def _is_rejected_real_scalar(value: Any) -> bool:
    return isinstance(value, _REJECTED_REAL_SCALAR_TYPES)


def _as_finite_float(value: Any, name: str) -> float:
    arr = np.asarray(value)
    if arr.ndim != 0 or arr.dtype == np.bool_ or arr.dtype.kind in "USbcMm":
        raise ValueError(f"{name} must be a finite scalar")
    scalar = arr.item()
    if _is_rejected_real_scalar(scalar):
        raise ValueError(f"{name} must be a finite scalar")
    try:
        result = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar")
    return result


def _as_nonnegative_time_delta(value: Any, name: str) -> float:
    arr = np.asarray(value)
    if arr.ndim != 0 or arr.dtype == np.bool_ or arr.dtype.kind in "USbcMm":
        raise ValueError(f"{name} must be nonnegative")
    scalar = arr.item()
    if _is_rejected_real_scalar(scalar):
        raise ValueError(f"{name} must be nonnegative")
    try:
        result = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be nonnegative") from exc
    if result < 0.0 or np.isnan(result):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _as_real_numeric_array(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == np.bool_ or arr.dtype.kind in "USbcMm":
        raise ValueError(f"{name} must contain real numeric values")
    if arr.dtype == object:
        for item in arr.reshape(-1):
            if _is_rejected_real_scalar(item):
                raise ValueError(f"{name} must contain real numeric values")
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc


def _as_summary_scalar(value: Any, name: str, *, allow_nan: bool = False) -> float:
    arr = np.asarray(value)
    if arr.ndim != 0 or arr.dtype == np.bool_ or arr.dtype.kind in "USbcMm":
        raise ValueError(f"{name} must be a real scalar")
    scalar = arr.item()
    if _is_rejected_real_scalar(scalar):
        raise ValueError(f"{name} must be a real scalar")
    try:
        result = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real scalar") from exc
    if np.isnan(result) and allow_nan:
        return result
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _as_nonnegative_summary_count(value: Any, name: str) -> float:
    result = _as_summary_scalar(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


_time_offset_module._as_finite_float = _as_finite_float
_time_offset_module._as_nonnegative_time_delta = _as_nonnegative_time_delta
_time_offset_module._as_real_numeric_array = _as_real_numeric_array
_time_offset_module._as_summary_scalar = _as_summary_scalar
_time_offset_module._as_nonnegative_summary_count = _as_nonnegative_summary_count

from .bias import (  # noqa: E402
    BiasTrainingExamples,
    SensorBiasCorrectionModel,
    fit_sensor_bias_correction,
    fit_sensor_bias_correction_from_examples,
    make_bias_training_examples,
)
from .time_offset import (  # noqa: E402
    TimeOffsetFitResult,
    _aggregate_std_metric,
    _validate_error_metric,
)
from .time_offset import aggregate_time_offset_sweeps as _aggregate_time_offset_sweeps
from .time_offset import (  # noqa: E402
    apply_time_offset,
    fit_time_offset,
    interpolate_reference_values,
    make_offset_grid,
    nearest_time_indices,
    time_offset_error_summary,
    time_offset_sweep,
)


def aggregate_time_offset_sweeps(
    sweeps: Iterable[Iterable[Mapping[str, float]]],
    *,
    metric: str = "rmse",
) -> list[dict[str, float]]:
    """Aggregate same-offset sweeps while preserving all summary metrics."""

    metric = _validate_error_metric(metric)
    materialized_sweeps = [list(sweep) for sweep in sweeps]
    rows = _aggregate_time_offset_sweeps(materialized_sweeps, metric=metric)
    if metric == "std":
        return rows

    by_offset: dict[float, list[Mapping[str, float]]] = {}
    for sweep in materialized_sweeps:
        for part in sweep:
            offset = _as_summary_scalar(part["time_offset_s"], "time_offset_s")
            by_offset.setdefault(offset, []).append(part)

    for row in rows:
        parts = by_offset.get(float(row["time_offset_s"]), ())
        counts = np.array(
            [
                _as_nonnegative_summary_count(part.get("count", 0.0), "count")
                for part in parts
            ],
            dtype=float,
        )
        values = np.array(
            [
                _as_summary_scalar(part.get("std", np.nan), "std", allow_nan=True)
                for part in parts
            ],
            dtype=float,
        )
        means = np.array(
            [
                _as_summary_scalar(part.get("mean", np.nan), "mean", allow_nan=True)
                for part in parts
            ],
            dtype=float,
        )
        row["std"] = _aggregate_std_metric(values, means, counts)
    return rows


_time_offset_module.aggregate_time_offset_sweeps = aggregate_time_offset_sweeps


__all__ = [
    "BiasTrainingExamples",
    "SensorBiasCorrectionModel",
    "TimeOffsetFitResult",
    "aggregate_time_offset_sweeps",
    "apply_time_offset",
    "fit_sensor_bias_correction",
    "fit_sensor_bias_correction_from_examples",
    "fit_time_offset",
    "interpolate_reference_values",
    "make_bias_training_examples",
    "make_offset_grid",
    "nearest_time_indices",
    "time_offset_error_summary",
    "time_offset_sweep",
]
