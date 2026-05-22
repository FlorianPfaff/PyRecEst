"""Compact diagnostic summaries for filter and tracker record streams."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

Record = Mapping[str, Any]


def build_diagnostic_summary(
    records: Sequence[Record],
    *,
    top_n: int = 20,
    window_s: float = 30.0,
    time_key: str = "time_s",
    residual_key: str = "residual_norm",
    error_key: str = "error",
    source_key: str = "source",
    track_id_key: str = "track_id",
    covariance_scale_key: str = "covariance_scale",
) -> dict[str, Any]:
    """Build a JSON-serializable diagnostic summary from mapping records."""

    if top_n < 1:
        raise ValueError("top_n must be positive")
    if window_s <= 0.0:
        raise ValueError("window_s must be positive")
    return {
        "schema_version": 1,
        "top_n": int(top_n),
        "window_s": float(window_s),
        "top_residuals": top_residuals(records, residual_key=residual_key, top_n=top_n),
        "track_switches": track_switch_summary(
            records, time_key=time_key, track_id_key=track_id_key, top_n=top_n
        ),
        "covariance_inflation": covariance_inflation_summary(
            records,
            scale_key=covariance_scale_key,
            source_key=source_key,
            top_n=top_n,
        ),
        "worst_time_windows": worst_time_windows(
            records,
            time_key=time_key,
            error_key=error_key,
            residual_key=residual_key,
            scale_key=covariance_scale_key,
            track_id_key=track_id_key,
            window_s=window_s,
            top_n=top_n,
        ),
    }


def top_residuals(
    records: Sequence[Record],
    *,
    residual_key: str = "residual_norm",
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """Return records with the largest finite residual values."""

    scored = []
    for index, record in enumerate(records):
        value = _optional_float(record.get(residual_key))
        if value is not None:
            scored.append((value, index, record))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [_json_record(record) for _, _, record in scored[:top_n]]


def track_switch_summary(
    records: Sequence[Record],
    *,
    time_key: str = "time_s",
    track_id_key: str = "track_id",
    top_n: int = 20,
) -> dict[str, Any]:
    """Summarize changes in finite track id over time."""

    ordered = _sort_by_time(records, time_key)
    finite: list[tuple[Any, Record]] = []
    for record in ordered:
        track_id = record.get(track_id_key)
        if track_id is not None and not _is_nan(track_id):
            finite.append((track_id, record))
    if not finite:
        return _empty_track_switch_summary()

    events: list[dict[str, Any]] = []
    transitions: Counter[tuple[Any, Any]] = Counter()
    previous_id: Any | None = None
    for current_id, record in finite:
        if previous_id is not None and current_id != previous_id:
            transitions[(previous_id, current_id)] += 1
            event = {"from_track_id": previous_id, "to_track_id": current_id}
            time_value = _optional_float(record.get(time_key))
            if time_value is not None:
                event["time_s"] = time_value
            events.append(_json_record(event))
        previous_id = current_id

    unique_ids = {track_id for track_id, _ in finite}
    return {
        "count": int(sum(transitions.values())),
        "updates_with_track_id": int(len(finite)),
        "unique_track_ids": int(len(unique_ids)),
        "first_track_id": _json_value(finite[0][0]),
        "last_track_id": _json_value(finite[-1][0]),
        "top_transitions": [
            {
                "from_track_id": _json_value(src),
                "to_track_id": _json_value(dst),
                "count": int(count),
            }
            for (src, dst), count in transitions.most_common(top_n)
        ],
        "events": events[:top_n],
    }


def covariance_inflation_summary(
    records: Sequence[Record],
    *,
    scale_key: str = "covariance_scale",
    source_key: str = "source",
    top_n: int = 20,
) -> dict[str, Any]:
    """Summarize updates whose covariance scale is greater than one."""

    inflated: list[tuple[float, Record]] = []
    by_source: Counter[str] = Counter()
    for record in records:
        scale = _optional_float(record.get(scale_key))
        if scale is not None and scale > 1.0:
            inflated.append((scale, record))
            by_source[str(record.get(source_key, "unknown"))] += 1
    if not inflated:
        return {
            "count": 0,
            "by_source": {},
            "mean_scale": None,
            "p95_scale": None,
            "max_scale": None,
            "top_scaled_updates": [],
        }
    scales = np.array([scale for scale, _ in inflated], dtype=float)
    inflated.sort(key=lambda item: item[0], reverse=True)
    return {
        "count": int(len(inflated)),
        "by_source": dict(sorted(by_source.items())),
        "mean_scale": float(np.mean(scales)),
        "p95_scale": float(np.percentile(scales, 95)),
        "max_scale": float(np.max(scales)),
        "top_scaled_updates": [_json_record(record) for _, record in inflated[:top_n]],
    }


def worst_time_windows(
    records: Sequence[Record],
    *,
    time_key: str = "time_s",
    error_key: str = "error",
    residual_key: str = "residual_norm",
    scale_key: str = "covariance_scale",
    track_id_key: str = "track_id",
    window_s: float = 30.0,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """Return windows with the largest RMSE for a supplied scalar error key."""

    if window_s <= 0.0:
        raise ValueError("window_s must be positive")
    grouped: dict[float, list[Record]] = {}
    for record in records:
        time_value = _optional_float(record.get(time_key))
        error = _optional_float(record.get(error_key))
        if time_value is None or error is None:
            continue
        start = math.floor(time_value / window_s) * window_s
        grouped.setdefault(float(start), []).append(record)

    rows: list[dict[str, Any]] = []
    for start, group in grouped.items():
        errors = np.array(
            [_optional_float(record.get(error_key)) for record in group], dtype=float
        )
        errors = errors[np.isfinite(errors)]
        if errors.size == 0:
            continue
        residuals = np.array(
            [
                value
                for value in (
                    _optional_float(record.get(residual_key)) for record in group
                )
                if value is not None
            ],
            dtype=float,
        )
        scales = np.array(
            [
                value
                for value in (
                    _optional_float(record.get(scale_key)) for record in group
                )
                if value is not None
            ],
            dtype=float,
        )
        rows.append(
            {
                "time_start_s": float(start),
                "time_end_s": float(start + window_s),
                "count": int(errors.size),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "mae": float(np.mean(np.abs(errors))),
                "p95": float(np.percentile(errors, 95)),
                "max": float(np.max(errors)),
                "mean_residual": (
                    None if residuals.size == 0 else float(np.mean(residuals))
                ),
                "covariance_inflation_count": (
                    int(np.sum(scales > 1.0)) if scales.size else 0
                ),
                "track_switch_count": track_switch_summary(
                    group, time_key=time_key, track_id_key=track_id_key, top_n=1
                )["count"],
            }
        )
    rows.sort(key=lambda item: item["rmse"], reverse=True)
    return rows[:top_n]


def _empty_track_switch_summary() -> dict[str, Any]:
    return {
        "count": 0,
        "updates_with_track_id": 0,
        "unique_track_ids": 0,
        "first_track_id": None,
        "last_track_id": None,
        "top_transitions": [],
        "events": [],
    }


def _sort_by_time(records: Sequence[Record], time_key: str) -> list[Record]:
    return sorted(
        records,
        key=lambda record: (
            (
                float("inf")
                if _optional_float(record.get(time_key)) is None
                else float(record.get(time_key))
            ),
        ),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _is_nan(value: Any) -> bool:
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _json_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_value(value) for key, value in record.items()}


def _json_value(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    return value


__all__ = [
    "build_diagnostic_summary",
    "covariance_inflation_summary",
    "top_residuals",
    "track_switch_summary",
    "worst_time_windows",
]
