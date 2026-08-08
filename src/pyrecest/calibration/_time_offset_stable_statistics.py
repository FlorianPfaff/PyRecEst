"""Overflow-safe residual norms and summary statistics for time-offset calibration."""

from __future__ import annotations

import sys
from functools import wraps
from typing import Any

import numpy as np

from . import time_offset as _time_offset

_ORIGINAL_ERROR_SUMMARY_ATTR = "_pyrecest_original_time_offset_error_summary"
_ORIGINAL_AGGREGATE_SUMMARY_ATTR = "_pyrecest_original_stable_aggregate_summary_metric"
_MARKER = "_pyrecest_stable_time_offset_statistics"


def _scaled_weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    """Return a weighted average without overflowing finite products or sums."""

    value_scale = float(np.max(np.abs(values), initial=0.0))
    if value_scale == 0.0:
        return 0.0
    weight_scale = float(np.max(weights, initial=0.0))
    normalized_weights = weights / weight_scale
    return float(
        value_scale
        * np.average(values / value_scale, weights=normalized_weights)
    )


def _stable_error_stats(
    offset_s: float, errors: np.ndarray, *, total_count: int
) -> dict[str, float]:
    """Summarize finite nonnegative errors after scale normalization."""

    errors = np.asarray(errors, dtype=float).reshape(-1)
    errors = errors[np.isfinite(errors)]
    if errors.size == 0:
        return {
            "time_offset_s": float(offset_s),
            "count": 0.0,
            "coverage": 0.0 if total_count else float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "rmse": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }

    scale = float(np.max(np.abs(errors), initial=0.0))
    scaled = errors if scale == 0.0 else errors / scale
    return {
        "time_offset_s": float(offset_s),
        "count": float(errors.size),
        "coverage": (
            float(errors.size / total_count) if total_count > 0 else float("nan")
        ),
        "mean": float(scale * np.mean(scaled)),
        "std": float(scale * np.std(scaled)),
        "rmse": float(scale * np.sqrt(np.mean(scaled * scaled))),
        "p95": float(scale * np.percentile(scaled, 95)),
        "max": float(np.max(errors)),
    }


def _stable_aggregate_summary_metric(
    key: str, values: np.ndarray, counts: np.ndarray
) -> float:
    """Aggregate finite metrics after normalizing values and weights."""

    if key == "p95":
        original = getattr(_time_offset, _ORIGINAL_AGGREGATE_SUMMARY_ATTR)
        return original(key, values, counts)

    valid = np.isfinite(values) & (counts > 0.0)
    if not valid.any():
        return float("nan")
    values = values[valid]
    weights = counts[valid]
    if key == "max":
        return float(np.max(values))
    if key == "rmse":
        scale = float(np.max(np.abs(values), initial=0.0))
        if scale == 0.0:
            return 0.0
        normalized_weights = weights / np.max(weights)
        return float(
            scale
            * np.sqrt(
                np.average((values / scale) ** 2, weights=normalized_weights)
            )
        )
    return _scaled_weighted_average(values, weights)


def _stable_aggregate_std_metric(
    stds: np.ndarray, means: np.ndarray, counts: np.ndarray
) -> float:
    """Pool means and standard deviations without squaring raw magnitudes."""

    valid = np.isfinite(stds) & np.isfinite(means) & (counts > 0.0)
    if not valid.any():
        return float("nan")
    stds = stds[valid]
    means = means[valid]
    weights = counts[valid]
    scale = float(
        max(
            np.max(np.abs(stds), initial=0.0),
            np.max(np.abs(means), initial=0.0),
        )
    )
    if scale == 0.0:
        return 0.0
    normalized_weights = weights / np.max(weights)
    scaled_means = means / scale
    scaled_stds = stds / scale
    pooled_mean = float(np.average(scaled_means, weights=normalized_weights))
    second_moment = float(
        np.average(
            scaled_stds * scaled_stds + scaled_means * scaled_means,
            weights=normalized_weights,
        )
    )
    return float(scale * np.sqrt(max(0.0, second_moment - pooled_mean**2)))


def _stable_time_offset_error_summary(
    measurement_times_s: np.ndarray,
    measurement_values: np.ndarray,
    reference_times_s: np.ndarray,
    reference_values: np.ndarray,
    offset_s: float | None,
    *,
    max_time_delta_s: float | None = None,
) -> dict[str, float]:
    """Compute residual norms without overflowing finite Euclidean distances."""

    offset = _time_offset._validate_time_offset(offset_s)
    measurement_values = _time_offset._as_real_numeric_array(
        measurement_values, "measurement_values"
    )
    if measurement_values.ndim == 1:
        measurement_values = measurement_values.reshape(-1, 1)
    elif measurement_values.ndim != 2:
        raise ValueError("measurement_values must be one- or two-dimensional")
    query_times = _time_offset.apply_time_offset(measurement_times_s, offset)
    if query_times.size != measurement_values.shape[0]:
        raise ValueError(
            "measurement_times_s length must match measurement_values rows"
        )
    reference_at_query, valid = _time_offset.interpolate_reference_values(
        reference_times_s,
        reference_values,
        query_times,
        max_time_delta_s=max_time_delta_s,
    )
    if measurement_values.shape[1] != reference_at_query.shape[1]:
        raise ValueError(
            "measurement_values and reference_values must have the same value dimension"
        )
    valid &= np.isfinite(measurement_values).all(axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        residuals = measurement_values[valid] - reference_at_query[valid]
        errors = np.hypot.reduce(np.abs(residuals), axis=1, initial=0.0)
    return _stable_error_stats(offset, errors, total_count=len(measurement_values))


def install_time_offset_stable_statistics_contract() -> None:
    """Install scale-safe residual and sweep statistics exactly once."""

    if not hasattr(_time_offset, _ORIGINAL_ERROR_SUMMARY_ATTR):
        setattr(
            _time_offset,
            _ORIGINAL_ERROR_SUMMARY_ATTR,
            _time_offset.time_offset_error_summary,
        )
    if not hasattr(_time_offset, _ORIGINAL_AGGREGATE_SUMMARY_ATTR):
        setattr(
            _time_offset,
            _ORIGINAL_AGGREGATE_SUMMARY_ATTR,
            _time_offset._aggregate_summary_metric,
        )
    if getattr(_time_offset.time_offset_error_summary, _MARKER, False):
        return

    original_summary = getattr(_time_offset, _ORIGINAL_ERROR_SUMMARY_ATTR)

    @wraps(original_summary)
    def checked_summary(*args: Any, **kwargs: Any) -> dict[str, float]:
        return _stable_time_offset_error_summary(*args, **kwargs)

    setattr(checked_summary, _MARKER, True)
    _time_offset.time_offset_error_summary = checked_summary
    _time_offset._error_stats = _stable_error_stats
    _time_offset._aggregate_summary_metric = _stable_aggregate_summary_metric
    _time_offset._aggregate_std_metric = _stable_aggregate_std_metric

    package_module = sys.modules.get(__package__)
    if package_module is not None:
        package_module.time_offset_error_summary = checked_summary
        package_module._aggregate_std_metric = _stable_aggregate_std_metric


__all__ = ["install_time_offset_stable_statistics_contract"]
