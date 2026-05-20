"""Standard diagnostic containers for filters, trackers, and samplers.

These dataclasses are intentionally lightweight. Algorithms can return them
through optional ``return_diagnostics=True`` code paths without introducing a
dependency on pandas, plotting libraries, or backend-specific array types.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class FilterDiagnostics:
    """Diagnostics commonly emitted by single-target Bayesian filters."""

    innovation: Any | None = None
    innovation_covariance: Any | None = None
    nis: float | None = None
    nees: float | None = None
    log_likelihood: float | None = None
    covariance_trace: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow JSON-friendly dictionary where possible."""
        return asdict(self)


@dataclass(slots=True)
class ParticleDiagnostics:
    """Diagnostics commonly emitted by particle filters and samplers."""

    effective_sample_size: float | None = None
    resampled: bool | None = None
    resampling_count: int | None = None
    weight_entropy: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AssociationDiagnostics:
    """Diagnostics for association and multi-target tracking steps."""

    cost_matrix: Any | None = None
    gated_measurement_indices: list[int] = field(default_factory=list)
    selected_assignments: list[tuple[int, int]] = field(default_factory=list)
    birth_labels: list[Any] = field(default_factory=list)
    death_labels: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "AssociationDiagnostics",
    "FilterDiagnostics",
    "ParticleDiagnostics",
]
