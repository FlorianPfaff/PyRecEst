"""Non-mutating measurement-scoring result containers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MeasurementScore:
    """Diagnostics for a measurement batch under the current filter state.

    The score is computed without applying the update.  ``residual`` and
    ``innovation_covariance`` are sufficient to compute likelihood-style gates
    or normalized innovation statistics.  ``quadratic_form`` is the innovation
    Mahalanobis distance ``r.T @ S^{-1} @ r`` when an active measurement batch is
    available.

    ``active_measurement_indices`` uses indices in the original measurement
    batch.  When no active measurement is available, all matrix-valued fields are
    ``None`` and ``skipped_reason`` explains why no score was computed.
    """

    measurement_jacobian: Any | None
    predicted_measurements: Any | None
    innovation_covariance: Any | None
    residual: Any | None
    active_measurement_indices: Sequence[int]
    measurement_weights: Any | None = None
    quadratic_form: float | None = None
    skipped_reason: str | None = None

    def __post_init__(self) -> None:
        # Own caller-provided sequences so this frozen result cannot be mutated indirectly.
        object.__setattr__(
            self,
            "active_measurement_indices",
            tuple(self.active_measurement_indices),
        )

    @property
    def is_active(self) -> bool:
        """Return whether the score contains at least one active measurement."""

        return len(self.active_measurement_indices) > 0
