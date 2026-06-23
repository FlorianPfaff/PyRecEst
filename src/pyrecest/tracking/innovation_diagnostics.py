"""Innovation and normalized-innovation-squared diagnostics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from pyrecest.filters.linear_update_planning import (
    chi_square_gate_threshold,
    normalized_innovation_squared,
)


@dataclass(frozen=True)
class InnovationDiagnostic:
    """Diagnostics for one innovation/update decision."""

    measurement_dim: int
    nis: float | None = None
    residual_norm: float | None = None
    gate_threshold: float | None = None
    accepted: bool | None = None
    action: str | None = None
    source: str | None = None
    time: float | None = None
    residual: np.ndarray | None = None
    innovation_covariance: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def mahalanobis_distance(self) -> float | None:
        """Return ``sqrt(max(nis, 0))`` when NIS is available."""

        if self.nis is None:
            return None
        return float(np.sqrt(max(0.0, float(self.nis))))

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        """Return a JSON/CSV-friendly representation."""

        row = asdict(self)
        row["metadata"] = dict(self.metadata)
        row["mahalanobis_distance"] = self.mahalanobis_distance
        if include_arrays:
            row["residual"] = (
                None if self.residual is None else np.asarray(self.residual).tolist()
            )
            row["innovation_covariance"] = (
                None
                if self.innovation_covariance is None
                else np.asarray(self.innovation_covariance).tolist()
            )
        else:
            row.pop("residual", None)
            row.pop("innovation_covariance", None)
        return row


@dataclass(frozen=True)
class InnovationSummary:
    """Aggregate innovation diagnostics for one group."""

    group: str
    count: int
    accepted_count: int
    rejected_count: int
    acceptance_rate: float | None
    nis_mean: float | None
    nis_median: float | None
    nis_p95: float | None
    nis_max: float | None
    residual_norm_mean: float | None
    residual_norm_median: float | None
    residual_norm_p95: float | None
    residual_norm_max: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/CSV-friendly representation."""

        return asdict(self)


def innovation_gate_threshold(
    probability: float | None,
    measurement_dim: int,
) -> float | None:
    """Return a chi-square NIS gate threshold for ``measurement_dim``."""

    return chi_square_gate_threshold(probability, measurement_dim)


def innovation_diagnostic(
    residual: np.ndarray,
    innovation_covariance: np.ndarray,
    *,
    gate_probability: float | None = None,
    gate_threshold: float | None = None,
    accepted: bool | None = None,
    action: str | None = None,
    source: str | None = None,
    time: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> InnovationDiagnostic:
    """Compute NIS and residual diagnostics from one innovation."""

    residual_array = np.asarray(residual, dtype=float).reshape(-1)
    innovation_covariance_array = np.asarray(innovation_covariance, dtype=float)
    if innovation_covariance_array.shape != (residual_array.size, residual_array.size):
        raise ValueError("innovation_covariance must match residual dimension")
    if (
        not np.isfinite(residual_array).all()
        or not np.isfinite(innovation_covariance_array).all()
    ):
        raise ValueError("innovation inputs must be finite")
    resolved_threshold = _validate_optional_positive_scalar(
        gate_threshold,
        "gate_threshold",
    )
    if resolved_threshold is None:
        resolved_threshold = innovation_gate_threshold(
            gate_probability, residual_array.size
        )
    nis = float(
        normalized_innovation_squared(residual_array, innovation_covariance_array)
    )
    inferred_accepted = accepted
    if inferred_accepted is None and resolved_threshold is not None:
        inferred_accepted = bool(nis <= resolved_threshold)
    return InnovationDiagnostic(
        measurement_dim=int(residual_array.size),
        nis=nis,
        residual_norm=float(np.linalg.norm(residual_array)),
        gate_threshold=(
            None if resolved_threshold is None else float(resolved_threshold)
        ),
        accepted=None if inferred_accepted is None else bool(inferred_accepted),
        action=action,
        source=source,
        time=None if time is None else float(time),
        residual=residual_array.copy(),
        innovation_covariance=innovation_covariance_array.copy(),
        metadata={} if metadata is None else dict(metadata),
    )


def linear_innovation_diagnostic(
    *,
    mean: np.ndarray,
    covariance: np.ndarray,
    measurement: np.ndarray,
    measurement_matrix: np.ndarray,
    measurement_covariance: np.ndarray,
    gate_probability: float | None = None,
    gate_threshold: float | None = None,
    accepted: bool | None = None,
    action: str | None = None,
    source: str | None = None,
    time: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> InnovationDiagnostic:
    """Compute innovation diagnostics for a linear measurement ``z = Hx + v``."""

    state_mean = np.asarray(mean, dtype=float).reshape(-1)
    state_covariance = np.asarray(covariance, dtype=float)
    measurement_vector = np.asarray(measurement, dtype=float).reshape(-1)
    observation = np.asarray(measurement_matrix, dtype=float)
    measurement_noise = np.asarray(measurement_covariance, dtype=float)
    if state_covariance.shape != (state_mean.size, state_mean.size):
        raise ValueError("covariance must have shape (state_dim, state_dim)")
    if observation.shape != (measurement_vector.size, state_mean.size):
        raise ValueError(
            "measurement_matrix must have shape (measurement_dim, state_dim)"
        )
    if measurement_noise.shape != (measurement_vector.size, measurement_vector.size):
        raise ValueError(
            "measurement_covariance must have shape (measurement_dim, measurement_dim)"
        )
    residual = measurement_vector - observation @ state_mean
    innovation_covariance = (
        observation @ state_covariance @ observation.T + measurement_noise
    )
    return innovation_diagnostic(
        residual,
        innovation_covariance,
        gate_probability=gate_probability,
        gate_threshold=gate_threshold,
        accepted=accepted,
        action=action,
        source=source,
        time=time,
        metadata=metadata,
    )


def diagnostic_from_record(
    record: Mapping[str, Any],
    *,
    source_key: str = "source",
    time_key: str = "time_s",
    action_key: str = "update_action",
    accepted_key: str = "accepted",
    nis_key: str = "nis",
    residual_norm_key: str = "residual_norm_m",
    measurement_dim_key: str = "measurement_dim",
    gate_threshold_key: str = "gate_threshold",
) -> InnovationDiagnostic:
    """Create an innovation diagnostic from a serialized tracking record."""

    measurement_dim_value = record.get(measurement_dim_key)
    if measurement_dim_value is None:
        residual = record.get("residual")
        if residual is not None:
            measurement_dim_value = np.asarray(residual).reshape(-1).size
        else:
            measurement_dim_value = 0
    measurement_dim = _nonnegative_integer(measurement_dim_value, measurement_dim_key)
    accepted = record.get(accepted_key)
    excluded = {
        source_key,
        time_key,
        action_key,
        accepted_key,
        nis_key,
        residual_norm_key,
        measurement_dim_key,
        gate_threshold_key,
    }
    return InnovationDiagnostic(
        measurement_dim=measurement_dim,
        nis=_optional_float(record.get(nis_key)),
        residual_norm=_optional_float(record.get(residual_norm_key)),
        gate_threshold=_optional_float(record.get(gate_threshold_key)),
        accepted=_optional_bool(accepted),
        action=None if record.get(action_key) is None else str(record.get(action_key)),
        source=None if record.get(source_key) is None else str(record.get(source_key)),
        time=_optional_float(record.get(time_key)),
        metadata={key: value for key, value in record.items() if key not in excluded},
    )


def diagnostics_from_records(
    records: Iterable[Mapping[str, Any]], **kwargs: Any
) -> list[InnovationDiagnostic]:
    """Create innovation diagnostics from serialized tracking records."""

    return [diagnostic_from_record(record, **kwargs) for record in records]


def diagnostics_to_dicts(
    diagnostics: Iterable[InnovationDiagnostic],
    *,
    include_arrays: bool = False,
) -> list[dict[str, Any]]:
    """Convert diagnostics to JSON/CSV-friendly dictionaries."""

    return [
        diagnostic.to_dict(include_arrays=include_arrays) for diagnostic in diagnostics
    ]


def summarize_innovation_diagnostics(
    diagnostics: Iterable[InnovationDiagnostic],
    *,
    group_by: str | None = "source",
) -> list[InnovationSummary]:
    """Summarize innovation diagnostics globally or by source/action."""

    diagnostics_list = list(diagnostics)
    if group_by is None:
        return [_summarize_group("all", diagnostics_list)]
    groups: dict[str, list[InnovationDiagnostic]] = {}
    for diagnostic in diagnostics_list:
        if group_by == "source":
            key = diagnostic.source or "unknown"
        elif group_by == "action":
            key = diagnostic.action or "unknown"
        elif group_by == "accepted":
            key = str(diagnostic.accepted)
        else:
            value = diagnostic.metadata.get(group_by)
            key = "unknown" if value is None else str(value)
        groups.setdefault(key, []).append(diagnostic)
    return [_summarize_group(key, items) for key, items in sorted(groups.items())]


def summaries_to_dicts(summaries: Iterable[InnovationSummary]) -> list[dict[str, Any]]:
    """Convert innovation summaries to JSON/CSV-friendly dictionaries."""

    return [summary.to_dict() for summary in summaries]


def _summarize_group(
    group: str, diagnostics: list[InnovationDiagnostic]
) -> InnovationSummary:
    accepted_values = [
        item.accepted for item in diagnostics if item.accepted is not None
    ]
    accepted_count = int(sum(bool(value) for value in accepted_values))
    rejected_count = int(sum(not bool(value) for value in accepted_values))
    acceptance_rate = (
        float(accepted_count / len(accepted_values)) if accepted_values else None
    )
    nis_values = np.asarray(
        [
            item.nis
            for item in diagnostics
            if item.nis is not None and np.isfinite(item.nis)
        ],
        dtype=float,
    )
    residual_values = np.asarray(
        [
            item.residual_norm
            for item in diagnostics
            if item.residual_norm is not None and np.isfinite(item.residual_norm)
        ],
        dtype=float,
    )
    return InnovationSummary(
        group=str(group),
        count=int(len(diagnostics)),
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        acceptance_rate=acceptance_rate,
        nis_mean=_mean_or_none(nis_values),
        nis_median=_percentile_or_none(nis_values, 50.0),
        nis_p95=_percentile_or_none(nis_values, 95.0),
        nis_max=_max_or_none(nis_values),
        residual_norm_mean=_mean_or_none(residual_values),
        residual_norm_median=_percentile_or_none(residual_values, 50.0),
        residual_norm_p95=_percentile_or_none(residual_values, 95.0),
        residual_norm_max=_max_or_none(residual_values),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _nonnegative_integer(value: Any, name: str) -> int:
    try:
        value_array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if value_array.shape != () or value_array.dtype == np.bool_:
        raise ValueError(f"{name} must be a non-negative integer")
    scalar = value_array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer")
    if isinstance(scalar, (int, np.integer)):
        parsed = int(scalar)
    else:
        try:
            scalar_float = float(scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a non-negative integer") from exc
        if not np.isfinite(scalar_float) or not scalar_float.is_integer():
            raise ValueError(f"{name} must be a non-negative integer")
        parsed = int(scalar_float)
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "nan", "none", "null"}:
            return None
        if normalized in {"1", "true", "t", "yes", "y"}:
            return True
        if normalized in {"0", "false", "f", "no", "n"}:
            return False
        raise ValueError("accepted must be a boolean-like value")

    try:
        value_array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("accepted must be a boolean-like value") from exc
    if value_array.shape != ():
        raise ValueError("accepted must be a scalar boolean-like value")
    scalar = value_array.item()
    if isinstance(scalar, (bool, np.bool_)):
        return bool(scalar)
    try:
        parsed = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("accepted must be a boolean-like value") from exc
    if not np.isfinite(parsed):
        return None
    if parsed == 0.0:
        return False
    if parsed == 1.0:
        return True
    raise ValueError("accepted must be a boolean-like value")


def _validate_optional_positive_scalar(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        value_array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive scalar") from exc
    if value_array.shape != () or value_array.dtype == np.bool_:
        raise ValueError(f"{name} must be a finite positive scalar")
    scalar = value_array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite positive scalar")
    try:
        parsed = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite positive scalar") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return parsed


def _mean_or_none(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.mean(values))


def _max_or_none(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.max(values))


def _percentile_or_none(values: np.ndarray, percentile: float) -> float | None:
    return None if values.size == 0 else float(np.percentile(values, percentile))


__all__ = [
    "InnovationDiagnostic",
    "InnovationSummary",
    "chi_square_gate_threshold",
    "diagnostic_from_record",
    "diagnostics_from_records",
    "diagnostics_to_dicts",
    "innovation_diagnostic",
    "innovation_gate_threshold",
    "linear_innovation_diagnostic",
    "normalized_innovation_squared",
    "summaries_to_dicts",
    "summarize_innovation_diagnostics",
]
