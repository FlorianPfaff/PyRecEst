"""Overflow-safe contracts for time-offset calibration error metrics."""

from __future__ import annotations

import sys
from functools import wraps
from typing import Any

import numpy as np

from . import time_offset as _time_offset

_ORIGINAL_ERROR_STATS_ATTR = "_pyrecest_original_time_offset_error_stats"
_ORIGINAL_ERROR_SUMMARY_ATTR = "_pyrecest_original_time_offset_error_summary"


def _stable_row_norms(values: np.ndarray) -> np.ndarray:
    """Return row-wise Euclidean norms without squaring large magnitudes."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be two-dimensional")
    if values.shape[1] == 0:
        return np.zeros(values.shape[0], dtype=float)

    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        scales = np.max(np.abs(values), axis=1)
        norms = np.array(scales, copy=True)
        finite_nonzero = np.isfinite(scales) & (scales > 0.0)
        if finite_nonzero.any():
            scaled = values[finite_nonzero] / scales[finite_nonzero, None]
            norms[finite_nonzero] = scales[finite_nonzero] * np.sqrt(
                np.sum(scaled * scaled, axis=1)
            )
        norms[scales == 0.0] = 0.0
    return norms


def _stable_error_stats(
    offset_s: float, errors: np.ndarray, *, total_count: int
) -> dict[str, float]:
    """Compute error summaries without overflow in moments or RMS values."""

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

    scale = float(np.max(np.abs(errors)))
    if scale == 0.0:
        mean = std = rmse = 0.0
    else:
        with np.errstate(
            over="ignore", under="ignore", invalid="ignore", divide="ignore"
        ):
            scaled = errors / scale
            mean = float(scale * np.mean(scaled))
            std = float(scale * np.std(scaled))
            rmse = float(scale * np.sqrt(np.mean(scaled * scaled)))

    return {
        "time_offset_s": float(offset_s),
        "count": float(errors.size),
        "coverage": (
            float(errors.size / total_count) if total_count > 0 else float("nan")
        ),
        "mean": mean,
        "std": std,
        "rmse": rmse,
        "p95": float(np.percentile(errors, 95)),
        "max": float(np.max(errors)),
    }


def _stable_time_offset_error_summary(
    measurement_times_s: np.ndarray,
    measurement_values: np.ndarray,
    reference_times_s: np.ndarray,
    reference_values: np.ndarray,
    offset_s: float | None,
    *,
    max_time_delta_s: float | None = None,
) -> dict[str, float]:
    """Evaluate offset errors while preserving large finite residuals."""

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
    errors = _stable_row_norms(residuals)
    return _time_offset._error_stats(
        offset, errors, total_count=len(measurement_values)
    )


def install_time_offset_metric_stability_contract() -> None:
    """Install overflow-safe time-offset residual and summary calculations."""

    if not hasattr(_time_offset, _ORIGINAL_ERROR_STATS_ATTR):
        setattr(
            _time_offset,
            _ORIGINAL_ERROR_STATS_ATTR,
            _time_offset._error_stats,
        )
    if not hasattr(_time_offset, _ORIGINAL_ERROR_SUMMARY_ATTR):
        setattr(
            _time_offset,
            _ORIGINAL_ERROR_SUMMARY_ATTR,
            _time_offset.time_offset_error_summary,
        )

    original = getattr(_time_offset, _ORIGINAL_ERROR_SUMMARY_ATTR)
    current = _time_offset.time_offset_error_summary
    if getattr(current, "_pyrecest_overflow_safe", False):
        checked = current
    else:

        @wraps(original)
        def checked(
            measurement_times_s: np.ndarray,
            measurement_values: np.ndarray,
            reference_times_s: np.ndarray,
            reference_values: np.ndarray,
            offset_s: float | None,
            *,
            max_time_delta_s: float | None = None,
        ) -> dict[str, float]:
            return _stable_time_offset_error_summary(
                measurement_times_s,
                measurement_values,
                reference_times_s,
                reference_values,
                offset_s,
                max_time_delta_s=max_time_delta_s,
            )

        setattr(checked, "_pyrecest_overflow_safe", True)
        _time_offset.time_offset_error_summary = checked

    _time_offset._error_stats = _stable_error_stats

    package_module: Any = sys.modules.get(__package__)
    if package_module is not None and hasattr(
        package_module, "time_offset_error_summary"
    ):
        package_module.time_offset_error_summary = checked
