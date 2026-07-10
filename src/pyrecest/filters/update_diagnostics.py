"""Reusable diagnostics for measurement-update based filters and trackers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from operator import index as operator_index
from typing import Any

_TEXT_SEQUENCE_TYPES = (str, bytes, bytearray)
_ACTIVE_INDICES_MESSAGE = (
    "active_measurement_indices must be a sequence of non-negative integers"
)


@dataclass(frozen=True)
class MeasurementUpdateDiagnostics:
    """Diagnostics captured from one measurement update.

    The class intentionally stores backend arrays as opaque objects so it can be
    reused by NumPy, PyTorch, and JAX-backed filters.  It standardizes the fields
    that are useful for gating, logging, and explaining why a measurement batch
    was skipped or accepted without imposing a specific distribution class.
    """

    active_measurement_indices: Sequence[int] | None = ()
    measurement_count: int | None = None
    measurement_weights: Any = None
    residual: Any = None
    innovation_covariance: Any = None
    quadratic_form: float | None = None
    skipped_reason: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self):
        indices = _normalize_active_measurement_indices(self.active_measurement_indices)
        object.__setattr__(self, "active_measurement_indices", indices)
        if self.measurement_count is not None:
            measurement_count = _as_nonnegative_integer(
                self.measurement_count,
                "measurement_count",
            )
            if indices and max(indices) >= measurement_count:
                raise ValueError(
                    "active_measurement_indices must be smaller than measurement_count"
                )
            object.__setattr__(self, "measurement_count", measurement_count)
        object.__setattr__(
            self, "skipped_reason", _normalize_skipped_reason(self.skipped_reason)
        )
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))

    @property
    def active_measurement_count(self) -> int:
        """Return the number of measurements that contributed to the update."""
        if self.active_measurement_indices is None:
            return 0
        return len(self.active_measurement_indices)

    @property
    def updated(self) -> bool:
        """Return whether the update used at least one active measurement."""
        return self.skipped_reason is None and self.active_measurement_count > 0

    @classmethod
    def skipped(
        cls,
        reason: str,
        *,
        measurement_count: int | None = None,
        measurement_weights: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "MeasurementUpdateDiagnostics":
        """Construct diagnostics for an update that intentionally did nothing."""
        return cls(
            active_measurement_indices=(),
            measurement_count=measurement_count,
            measurement_weights=measurement_weights,
            skipped_reason=reason,
            metadata=metadata,
        )


def _normalize_active_measurement_indices(
    values: Sequence[int] | None,
) -> tuple[int, ...]:
    if values is None:
        return ()
    if isinstance(values, _TEXT_SEQUENCE_TYPES):
        raise ValueError(_ACTIVE_INDICES_MESSAGE)
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise ValueError(_ACTIVE_INDICES_MESSAGE) from exc
    indices = tuple(
        _as_nonnegative_integer(value, "active_measurement_indices")
        for value in iterator
    )
    if len(set(indices)) != len(indices):
        raise ValueError(
            "active_measurement_indices must not contain duplicate indices"
        )
    return indices


def _normalize_skipped_reason(skipped_reason: str | None) -> str | None:
    if skipped_reason is None:
        return None
    if not isinstance(skipped_reason, str):
        raise ValueError("skipped_reason must be a string or None")
    if not skipped_reason:
        raise ValueError("skipped_reason must not be empty")
    return skipped_reason


def _normalize_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping or None")
    return dict(metadata)


def _as_nonnegative_integer(value: Any, name: str) -> int:
    message = f"{name} must be a non-negative integer"
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        parsed = operator_index(value)
    except TypeError as exc:
        raise ValueError(message) from exc
    if parsed < 0:
        raise ValueError(message)
    return int(parsed)
