"""Standard diagnostic containers for filters, trackers, and samplers.

These dataclasses are intentionally lightweight. Algorithms can return them
through optional ``return_diagnostics=True`` code paths without introducing a
dependency on pandas, plotting libraries, or backend-specific array types.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import log
from typing import Any


def _coerce_weight_values(weights: Any) -> list[float]:
    """Return backend-independent Python floats from an array-like weight vector."""
    try:
        from pyrecest.backend import to_numpy

        weights = to_numpy(weights)
    except Exception:  # pragma: no cover - best-effort fallback for foreign arrays
        pass

    if hasattr(weights, "tolist"):
        weights = weights.tolist()
    if isinstance(weights, int | float):
        return [float(weights)]
    return [float(weight) for weight in weights]


class _DiagnosticsMappingMixin:
    """Small mapping compatibility layer for legacy diagnostics dictionaries."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()

    def __getitem__(self, key: str) -> Any:
        try:
            return self.to_dict()[key]
        except KeyError as exc:
            raise KeyError(key) from exc

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
            return
        metadata = getattr(self, "metadata")
        metadata[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def items(self) -> Any:
        return self.to_dict().items()


@dataclass(slots=True)
class FilterDiagnostics(_DiagnosticsMappingMixin):
    """Diagnostics commonly emitted by single-target Bayesian filters."""

    innovation: Any | None = None
    innovation_covariance: Any | None = None
    residual: Any | None = None
    nis: float | None = None
    nees: float | None = None
    log_likelihood: float | None = None
    covariance_trace: float | None = None
    scale: float | None = None
    action: str | None = None
    accepted: bool | None = None
    robust_update: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "FilterDiagnostics":
        known = {field_name for field_name in cls.__dataclass_fields__}
        values = {key: value for key, value in mapping.items() if key in known}
        metadata = dict(mapping.get("metadata", {}))
        metadata.update(
            {key: value for key, value in mapping.items() if key not in known}
        )
        values["metadata"] = metadata
        return cls(**values)


@dataclass(slots=True)
class ParticleDiagnostics(_DiagnosticsMappingMixin):
    """Diagnostics commonly emitted by particle filters and samplers."""

    effective_sample_size: float | None = None
    resampled: bool | None = None
    resampling_count: int | None = None
    weight_entropy: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_weights(
        cls,
        weights: Any,
        *,
        resampled: bool | None = None,
        resampling_count: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ParticleDiagnostics":
        """Build particle diagnostics from normalized or unnormalized weights."""
        values = _coerce_weight_values(weights)
        total = sum(values)
        if total <= 0.0:
            normalized = [0.0 for _ in values]
        else:
            normalized = [max(0.0, value / total) for value in values]
        squared_sum = sum(weight * weight for weight in normalized)
        effective_sample_size = 1.0 / squared_sum if squared_sum > 0.0 else 0.0
        entropy = -sum(weight * log(weight) for weight in normalized if weight > 0.0)
        return cls(
            effective_sample_size=effective_sample_size,
            resampled=resampled,
            resampling_count=resampling_count,
            weight_entropy=entropy,
            metadata={} if metadata is None else dict(metadata),
        )


@dataclass(slots=True)
class AssociationDiagnostics(_DiagnosticsMappingMixin):
    """Diagnostics for association and multi-target tracking steps."""

    cost_matrix: Any | None = None
    gated_measurement_indices: list[int] = field(default_factory=list)
    selected_assignments: list[tuple[int, int]] = field(default_factory=list)
    birth_labels: list[Any] = field(default_factory=list)
    death_labels: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "AssociationDiagnostics",
    "FilterDiagnostics",
    "ParticleDiagnostics",
]
