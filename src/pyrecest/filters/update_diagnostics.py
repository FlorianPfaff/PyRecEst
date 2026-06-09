"""Reusable diagnostics for measurement-update based filters and trackers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


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
        indices = (
            ()
            if self.active_measurement_indices is None
            else tuple(int(index) for index in self.active_measurement_indices)
        )
        object.__setattr__(self, "active_measurement_indices", indices)
        if self.measurement_count is not None and self.measurement_count < 0:
            raise ValueError("measurement_count must be non-negative when provided")
        metadata = {} if self.metadata is None else dict(self.metadata)
        object.__setattr__(self, "metadata", metadata)

    @property
    def active_measurement_count(self) -> int:
        """Return the number of measurements that contributed to the update."""
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
