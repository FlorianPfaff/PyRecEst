"""Generic timestamp-offset calibration utilities."""

from __future__ import annotations

import datetime as _datetime
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_ERROR_METRIC_NAMES = frozenset({"max", "mean", "p95", "rmse", "std"})
_ERROR_METRIC_MESSAGE = "metric must be one of 'max', 'mean', 'p95', 'rmse', or 'std'"
_TEMPORAL_SCALAR_TYPES = (
    np.datetime64,
    np.timedelta64,
    _datetime.date,
    _datetime.datetime,
    _datetime.timedelta,
)
_TEMPORAL_REPR_MARKERS = ("datetime64", "timedelta64")
_TIME_OFFSET_SUMMARY_FIELDS = frozenset(
    {
        "metric",
        "best_offset_s",
        "evaluated_offsets",
        "best_metric_value",
        "best_count",
    }
)


@dataclass(frozen=True)
class TimeOffsetFitResult:
    """Best timestamp correction selected from an offset sweep."""

    best_offset_s: float | None
    metric: str
    offsets_s: np.ndarray
    metric_values: np.ndarray
    counts: np.ndarray
    summaries: list[dict[str, float]] = field(default_factory=list)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        out = {
            key: value
            for key, value in dict(self.metadata).items()
            if key not in _TIME_OFFSET_SUMMARY_FIELDS
        }
        out.update(
            {
                "metric": self.metric,
                "best_offset_s": self.best_offset_s,
                "evaluated_offsets": int(len(self.offsets_s)),
            }
        )
        best_index = _best_metric_index(
            self.offsets_s,
            self.metric_values,
            self.counts,
            self.best_offset_s,
        )
        if best_index is not None:
            out["best_metric_value"] = float(self.metric_values[best_index])
            out["best_count"] = int(self.counts[best_index])
        return out


def _best_metric_index(
    offsets_s: np.ndarray,
    metric_values: np.ndarray,
    counts: np.ndarray,
    best_offset_s: float | None,
) -> int | None:
    if best_offset_s is None:
        return None
    offsets = np.asarray(offsets_s, dtype=float).reshape(-1)
    values = np.asarray(metric_values, dtype=float).reshape(-1)
    counts = np.asarray(counts, dtype=float).reshape(-1)
    if not (offsets.size == values.size == counts.size):
        return None
    finite = np.isfinite(values) & (counts > 0.0)
    if not finite.any():
        return None
    matching = finite & np.isclose(offsets, float(best_offset_s), rtol=0.0, atol=1e-12)
    candidates = np.flatnonzero(matching if matching.any() else finite)
    return int(candidates[int(np.nanargmin(values[candidates]))])


def _validate_error_metric(metric: Any) -> str:
    if not isinstance(metric, str):
        raise ValueError(_ERROR_METRIC_MESSAGE)
    normalized = metric.strip().lower()
    if normalized not in _ERROR_METRIC_NAMES:
        raise ValueError(_ERROR_METRIC_MESSAGE)
    return normalized


def _contains_temporal_values(arr: np.ndarray) -> bool:
    """Return true for native or object-wrapped NumPy temporal values."""
    if arr.dtype.kind in "Mm":
        return True
    if arr.dtype.kind != "O":
        return False
    if _has_temporal_repr_marker(arr):
        return True
    for item in arr.reshape(-1):
        if _is_temporal_value(item):
            return True
    return False


def _is_temporal_value(value: Any) -> bool:
    if isinstance(value, _TEMPORAL_SCALAR_TYPES):
        return True
    dtype = getattr(value, "dtype", None)
    if getattr(dtype, "kind", None) in ("M", "m"):
        return True
    if isinstance(value, np.ndarray):
        return _contains_temporal_values(value)
    return _has_temporal_repr_marker(value)


def _has_temporal_repr_marker(value: Any) -> bool:
    """Catch NumPy object temporal scalars whose iterated value lost dtype info."""
    value_repr = repr(value).lower()
    return any(marker in value_repr for marker in _TEMPORAL_REPR_MARKERS)


def _as_finite_float(value: Any, name: str) -> float:
    arr = np.asarray(value)
    if (
        arr.ndim != 0
        or arr.dtype == np.bool_
        or arr.dtype.kind == "O"
        or _contains_temporal_values(arr)
    ):
        raise ValueError(f"{name} must be a finite scalar")
    scalar = arr.item()
    if isinstance(scalar, (bool, np.bool_, str, bytes, bytearray)):
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
    if (
        arr.ndim != 0
        or arr.dtype == np.bool_
        or arr.dtype.kind == "O"
        or _contains_temporal_values(arr)
    ):
        raise ValueError(f"{name} must be nonnegative")
    scalar = arr.item()
    if isinstance(scalar, (bool, np.bool_, str, bytes, bytearray)):
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
    if (
        arr.dtype == np.bool_
        or arr.dtype.kind in "OUSbc"
        or _contains_temporal_values(arr)
    ):
        raise ValueError(f"{name} must contain real numeric values")
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc


def _as_summary_scalar(value: Any, name: str, *, allow_nan: bool = False) -> float:
    arr = np.asarray(value)
    if (
        arr.ndim != 0
        or arr.dtype == np.bool_
        or arr.dtype.kind in "OUSbc"
        or _contains_temporal_values(arr)
    ):
        raise ValueError(f"{name} must be a real scalar")
    scalar = arr.item()
    if isinstance(scalar, (bool, np.bool_, str, bytes, bytearray)):
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


def make_offset_grid(min_s: float, max_s: float, step_s: float) -> np.ndarray:
    min_s = _as_finite_float(min_s, "min_s")
    max_s = _as_finite_float(max_s, "max_s")
    step_s = _as_finite_float(step_s, "step_s")
    if step_s <= 0.0:
        raise ValueError("step_s must be positive")
    if max_s < min_s:
        raise ValueError("max_s must be greater than or equal to min_s")
    count = int(np.floor((max_s - min_s) / step_s)) + 1
    offsets = min_s + np.arange(count, dtype=float) * step_s
    if offsets.size == 0 or offsets[-1] < max_s - 1.0e-12:
        offsets = np.append(offsets, max_s)
    return offsets


def _validate_time_offset(offset_s: float | None) -> float:
    return 0.0 if offset_s is None else _as_finite_float(offset_s, "offset_s")


def apply_time_offset(times_s: np.ndarray, offset_s: float | None) -> np.ndarray:
    offset = _validate_time_offset(offset_s)
    return _as_real_numeric_array(times_s, "times_s") + offset


def _validate_max_time_delta(max_time_delta_s: float | None) -> float | None:
    return (
        None
        if max_time_delta_s is None
        else _as_nonnegative_time_delta(max_time_delta_s, "max_time_delta_s")
    )


def _finite_reference_rows(
    reference_times_s: np.ndarray, reference_values: np.ndarray | None = None
) -> np.ndarray:
    reference_times = _as_real_numeric_array(
        reference_times_s, "reference_times_s"
    ).reshape(-1)
    finite = np.isfinite(reference_times)
    if reference_values is not None:
        values = _as_real_numeric_array(reference_values, "reference_values")
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        finite &= np.isfinite(values).all(axis=1)
    return finite


def nearest_time_indices(
    reference_times_s: np.ndarray, query_times_s: np.ndarray
) -> np.ndarray:
    reference = _as_real_numeric_array(reference_times_s, "reference_times_s").reshape(
        -1
    )
    query = _as_real_numeric_array(query_times_s, "query_times_s").reshape(-1)
    finite_reference = _finite_reference_rows(reference)
    if not finite_reference.any():
        raise ValueError("reference_times_s must contain at least one finite value")
    nearest = np.full(query.shape, -1, dtype=int)
    finite_query = np.isfinite(query)
    if not finite_query.any():
        return nearest
    original_indices = np.flatnonzero(finite_reference)
    reference = reference[finite_reference]
    order = np.argsort(reference)
    sorted_reference = reference[order]
    finite_query_values = query[finite_query]
    insertion = np.searchsorted(sorted_reference, finite_query_values)
    right = np.clip(insertion, 0, sorted_reference.size - 1)
    left = np.clip(insertion - 1, 0, sorted_reference.size - 1)
    use_right = np.abs(sorted_reference[right] - finite_query_values) < np.abs(
        sorted_reference[left] - finite_query_values
    )
    nearest[finite_query] = original_indices[order[np.where(use_right, right, left)]]
    return nearest


def interpolate_reference_values(
    reference_times_s: np.ndarray,
    reference_values: np.ndarray,
    query_times_s: np.ndarray,
    *,
    max_time_delta_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    max_time_delta = _validate_max_time_delta(max_time_delta_s)
    reference_times = _as_real_numeric_array(
        reference_times_s, "reference_times_s"
    ).reshape(-1)
    reference_values = _as_real_numeric_array(reference_values, "reference_values")
    query_times = _as_real_numeric_array(query_times_s, "query_times_s").reshape(-1)
    if reference_values.ndim not in (1, 2):
        raise ValueError("reference_values must be one- or two-dimensional")
    if reference_values.ndim == 1:
        reference_values = reference_values.reshape(-1, 1)
    if reference_times.size != reference_values.shape[0]:
        raise ValueError("reference_times_s length must match reference_values rows")
    finite_reference = _finite_reference_rows(reference_times, reference_values)
    if np.count_nonzero(finite_reference) < 2:
        raise ValueError(
            "at least two finite reference rows are required for interpolation"
        )
    reference_times = reference_times[finite_reference]
    reference_values = reference_values[finite_reference]
    order = np.argsort(reference_times)
    reference_times = reference_times[order]
    reference_values = reference_values[order]
    interpolated = np.column_stack(
        [
            np.interp(query_times, reference_times, reference_values[:, dim])
            for dim in range(reference_values.shape[1])
        ]
    )
    valid = (
        np.isfinite(query_times)
        & (query_times >= reference_times[0])
        & (query_times <= reference_times[-1])
    )
    if max_time_delta is not None:
        nearest = nearest_time_indices(reference_times, query_times)
        valid &= np.abs(reference_times[nearest] - query_times) <= max_time_delta
    valid &= np.isfinite(interpolated).all(axis=1)
    return interpolated, valid


def time_offset_error_summary(
    measurement_times_s: np.ndarray,
    measurement_values: np.ndarray,
    reference_times_s: np.ndarray,
    reference_values: np.ndarray,
    offset_s: float | None,
    *,
    max_time_delta_s: float | None = None,
) -> dict[str, float]:
    offset = _validate_time_offset(offset_s)
    measurement_values = _as_real_numeric_array(
        measurement_values, "measurement_values"
    )
    if measurement_values.ndim == 1:
        measurement_values = measurement_values.reshape(-1, 1)
    elif measurement_values.ndim != 2:
        raise ValueError("measurement_values must be one- or two-dimensional")
    query_times = apply_time_offset(measurement_times_s, offset)
    if query_times.size != measurement_values.shape[0]:
        raise ValueError(
            "measurement_times_s length must match measurement_values rows"
        )
    reference_at_query, valid = interpolate_reference_values(
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
    errors = np.linalg.norm(
        measurement_values[valid] - reference_at_query[valid], axis=1
    )
    return _error_stats(offset, errors, total_count=len(measurement_values))


def time_offset_sweep(
    measurement_times_s: np.ndarray,
    measurement_values: np.ndarray,
    reference_times_s: np.ndarray,
    reference_values: np.ndarray,
    offsets_s: Iterable[float | None],
    *,
    max_time_delta_s: float | None = None,
) -> list[dict[str, float]]:
    return [
        time_offset_error_summary(
            measurement_times_s,
            measurement_values,
            reference_times_s,
            reference_values,
            offset,
            max_time_delta_s=max_time_delta_s,
        )
        for offset in offsets_s
    ]


def fit_time_offset(
    measurement_times_s: np.ndarray,
    measurement_values: np.ndarray,
    reference_times_s: np.ndarray,
    reference_values: np.ndarray,
    offsets_s: Iterable[float | None],
    *,
    metric: str = "rmse",
    max_time_delta_s: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TimeOffsetFitResult:
    metric = _validate_error_metric(metric)
    summaries = time_offset_sweep(
        measurement_times_s,
        measurement_values,
        reference_times_s,
        reference_values,
        offsets_s,
        max_time_delta_s=max_time_delta_s,
    )
    offsets = np.array([row["time_offset_s"] for row in summaries], dtype=float)
    values = np.array([row[metric] for row in summaries], dtype=float)
    counts = np.array([row.get("count", 0.0) for row in summaries], dtype=float)
    finite = np.isfinite(values) & (counts > 0)
    best = (
        None
        if not finite.any()
        else float(offsets[np.where(finite)[0][int(np.nanargmin(values[finite]))]])
    )
    return TimeOffsetFitResult(
        best_offset_s=best,
        metric=metric,
        offsets_s=offsets,
        metric_values=values,
        counts=counts.astype(int),
        summaries=summaries,
        metadata={} if metadata is None else dict(metadata),
    )


def aggregate_time_offset_sweeps(
    sweeps: Iterable[Iterable[Mapping[str, float]]], *, metric: str = "rmse"
) -> list[dict[str, float]]:
    metric = _validate_error_metric(metric)
    by_offset: dict[float, list[Mapping[str, float]]] = {}
    for sweep in sweeps:
        for row in sweep:
            offset = _as_summary_scalar(row["time_offset_s"], "time_offset_s")
            by_offset.setdefault(offset, []).append(row)
    rows: list[dict[str, float]] = []
    for offset, parts in sorted(by_offset.items()):
        counts = np.array(
            [
                _as_nonnegative_summary_count(part.get("count", 0.0), "count")
                for part in parts
            ],
            dtype=float,
        )
        row = {"time_offset_s": float(offset), "count": float(np.sum(counts))}
        for key in dict.fromkeys(("mean", "std", "rmse", "p95", "max", metric)):
            values = np.array(
                [
                    _as_summary_scalar(part.get(key, np.nan), str(key), allow_nan=True)
                    for part in parts
                ],
                dtype=float,
            )
            if key == "std":
                means = np.array(
                    [
                        _as_summary_scalar(
                            part.get("mean", np.nan), "mean", allow_nan=True
                        )
                        for part in parts
                    ],
                    dtype=float,
                )
                row[key] = _aggregate_std_metric(values, means, counts)
            else:
                row[key] = _aggregate_summary_metric(key, values, counts)
        rows.append(row)
    return rows


def _aggregate_summary_metric(
    key: str, values: np.ndarray, counts: np.ndarray
) -> float:
    valid = np.isfinite(values) & (counts > 0.0)
    if not valid.any():
        return float("nan")
    if key == "rmse":
        return float(np.sqrt(np.average(values[valid] ** 2, weights=counts[valid])))
    if key == "max":
        return float(np.max(values[valid]))
    return float(np.average(values[valid], weights=counts[valid]))


def _aggregate_std_metric(
    stds: np.ndarray, means: np.ndarray, counts: np.ndarray
) -> float:
    valid = np.isfinite(stds) & np.isfinite(means) & (counts > 0.0)
    if not valid.any():
        return float("nan")
    weights = counts[valid]
    pooled_mean = float(np.average(means[valid], weights=weights))
    second_moment = float(
        np.average(stds[valid] ** 2 + means[valid] ** 2, weights=weights)
    )
    return float(np.sqrt(max(0.0, second_moment - pooled_mean**2)))


def _error_stats(
    offset_s: float, errors: np.ndarray, *, total_count: int
) -> dict[str, float]:
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
    return {
        "time_offset_s": float(offset_s),
        "count": float(errors.size),
        "coverage": (
            float(errors.size / total_count) if total_count > 0 else float("nan")
        ),
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "p95": float(np.percentile(errors, 95)),
        "max": float(np.max(errors)),
    }


__all__ = [
    "TimeOffsetFitResult",
    "aggregate_time_offset_sweeps",
    "apply_time_offset",
    "fit_time_offset",
    "interpolate_reference_values",
    "make_offset_grid",
    "nearest_time_indices",
    "time_offset_error_summary",
    "time_offset_sweep",
]
