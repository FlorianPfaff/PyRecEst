"""Calibration helpers for asynchronous sensor-fusion workflows."""

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from . import time_offset as _time_offset_module
from .bias import (
    BiasTrainingExamples,
    SensorBiasCorrectionModel,
    fit_sensor_bias_correction,
    fit_sensor_bias_correction_from_examples,
    make_bias_training_examples,
)
from .time_offset import (
    TimeOffsetFitResult,
    _aggregate_std_metric,
    _as_nonnegative_summary_count,
    _as_summary_scalar,
    _validate_error_metric,
)
from .time_offset import aggregate_time_offset_sweeps as _aggregate_time_offset_sweeps
from .time_offset import (
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
