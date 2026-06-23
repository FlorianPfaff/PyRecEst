"""Standard diagnostic containers for filters, trackers, and samplers.

These dataclasses are intentionally lightweight. Algorithms can return them
through optional ``return_diagnostics=True`` code paths without introducing a
dependency on pandas, plotting libraries, or backend-specific array types.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from math import isfinite, log
from typing import Any, Literal

EvidenceSupportType = Literal[
    "exact_full_grid",
    "exact_sparse",
    "truncated_lower_bound",
    "approximate_particle",
    "unknown",
]

_EVIDENCE_SUPPORT_TYPES = set(EvidenceSupportType.__args__)


def _coerce_weight_values(weights: Any) -> list[float]:
    """Return backend-independent Python floats from an array-like weight vector."""
    try:
        from pyrecest.backend import to_numpy

        weights = to_numpy(weights)
    except Exception:  # pragma: no cover  # pylint: disable=broad-exception-caught
        pass

    if hasattr(weights, "tolist"):
        weights = weights.tolist()
    if isinstance(weights, int | float):
        return [float(weights)]
    return [float(weight) for weight in weights]


def _coerce_numeric_values(values: Any) -> list[float]:
    """Return a flat list of finite-compatible numeric values."""
    if values is None:
        return []
    if isinstance(values, bool | int | float):
        return [float(values)]
    if isinstance(values, str | bytes):
        return []
    if isinstance(values, list | tuple):
        out: list[float] = []
        for value in values:
            out.extend(_coerce_numeric_values(value))
        return out
    try:
        from pyrecest.backend import to_numpy

        values = to_numpy(values)
    except Exception:  # pragma: no cover - best-effort fallback for foreign arrays
        pass

    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, bool | int | float):
        return [float(values)]
    if isinstance(values, str | bytes):
        return []

    out: list[float] = []
    try:
        iterator = iter(values)
    except TypeError:
        try:
            return [float(values)]
        except (TypeError, ValueError, OverflowError):
            return []
    for value in iterator:
        out.extend(_coerce_numeric_values(value))
    return out


def _coerce_bool_values(values: Any) -> list[bool]:
    """Return a flat list of Boolean values."""
    if values is None:
        return []
    if isinstance(values, bool | int | float):
        return [bool(values)]
    if isinstance(values, str | bytes):
        return []
    if isinstance(values, list | tuple):
        out: list[bool] = []
        for value in values:
            out.extend(_coerce_bool_values(value))
        return out
    try:
        from pyrecest.backend import to_numpy

        values = to_numpy(values)
    except Exception:  # pragma: no cover - best-effort fallback for foreign arrays
        pass

    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, bool | int | float):
        return [bool(values)]
    if isinstance(values, str | bytes):
        return []

    out: list[bool] = []
    try:
        iterator = iter(values)
    except TypeError:
        return []
    for value in iterator:
        out.extend(_coerce_bool_values(value))
    return out


def _finite_mean(values: list[float]) -> float | None:
    valid = [value for value in values if isfinite(value)]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _finite_min(values: list[float]) -> float | None:
    valid = [value for value in values if isfinite(value)]
    if not valid:
        return None
    return min(valid)


def _finite_last(values: list[float]) -> float | None:
    for value in reversed(values):
        if isfinite(value):
            return value
    return None


def _normalized_nonnegative_weights(values: list[float]) -> list[float]:
    if any(not isfinite(value) for value in values):
        raise ValueError("Particle weights must be finite.")
    nonnegative_values = [max(0.0, value) for value in values]
    total = sum(nonnegative_values)
    if total <= 0.0:
        return [0.0 for _ in nonnegative_values]
    return [value / total for value in nonnegative_values]


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


@dataclass(frozen=True, slots=True)
class EvidenceSupport:
    """Comparability metadata for a reported log evidence or score.

    Recursive-estimation benchmarks often mix exact marginal likelihoods,
    evidence lower bounds, finite-support exact computations, particle
    approximations, and heuristic scores.  This lightweight container records
    how a reported evidence value was produced so downstream tables can avoid
    normalizing or ranking incomparable rows as if they were exact evidences.

    Parameters
    ----------
    support_type:
        Coarse evidence-support class.  The values are deliberately generic and
        not tied to a particular application domain.
    comparable:
        ``True`` only when the evidence is intended to be directly comparable to
        exact evidence values for the same data and support definition.
    lower_bound:
        ``True`` when the reported value is a lower bound or truncation-based
        audit value rather than the exact marginal likelihood.
    diagnostics:
        Optional algorithm-specific metadata such as support size, retained mass,
        particle count, or the reason a value is not comparable.
    """

    support_type: EvidenceSupportType = "unknown"
    comparable: bool = False
    lower_bound: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.support_type not in _EVIDENCE_SUPPORT_TYPES:
            raise ValueError(
                f"unsupported evidence support type {self.support_type!r}; "
                f"expected one of {sorted(_EVIDENCE_SUPPORT_TYPES)}"
            )
        if not isinstance(self.diagnostics, dict):
            object.__setattr__(self, "diagnostics", dict(self.diagnostics))
        if self.support_type == "truncated_lower_bound" and not self.lower_bound:
            object.__setattr__(self, "lower_bound", True)

    @property
    def headline_comparable(self) -> bool:
        """Return whether the value is safe for headline exact-evidence rankings."""

        return bool(self.comparable and not self.lower_bound)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary suitable for JSON or dataframe rows."""

        return {
            "support_type": self.support_type,
            "comparable": bool(self.comparable),
            "lower_bound": bool(self.lower_bound),
            "headline_comparable": self.headline_comparable,
            "diagnostics": dict(self.diagnostics),
        }

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()

    def __getitem__(self, key: str) -> Any:
        try:
            return self.to_dict()[key]
        except KeyError as exc:
            raise KeyError(key) from exc

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def items(self) -> Any:
        return self.to_dict().items()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "EvidenceSupport":
        """Build support metadata from a mapping, preserving unknown keys."""

        known = {"support_type", "comparable", "lower_bound", "diagnostics"}
        diagnostics = dict(mapping.get("diagnostics", {}))
        diagnostics.update(
            {key: value for key, value in mapping.items() if key not in known}
        )
        return cls(
            support_type=mapping.get("support_type", "unknown"),
            comparable=bool(mapping.get("comparable", False)),
            lower_bound=bool(mapping.get("lower_bound", False)),
            diagnostics=diagnostics,
        )

    @classmethod
    def exact_full_grid(
        cls, diagnostics: Mapping[str, Any] | None = None
    ) -> "EvidenceSupport":
        """Evidence computed exactly over the complete declared finite grid."""

        return cls(
            support_type="exact_full_grid",
            comparable=True,
            lower_bound=False,
            diagnostics={} if diagnostics is None else dict(diagnostics),
        )

    @classmethod
    def exact_sparse(
        cls, diagnostics: Mapping[str, Any] | None = None
    ) -> "EvidenceSupport":
        """Evidence computed exactly over a declared sparse finite support."""

        return cls(
            support_type="exact_sparse",
            comparable=True,
            lower_bound=False,
            diagnostics={} if diagnostics is None else dict(diagnostics),
        )

    @classmethod
    def truncated_lower_bound(
        cls, diagnostics: Mapping[str, Any] | None = None
    ) -> "EvidenceSupport":
        """Evidence lower bound from a truncated/candidate-pruned support."""

        return cls(
            support_type="truncated_lower_bound",
            comparable=False,
            lower_bound=True,
            diagnostics={} if diagnostics is None else dict(diagnostics),
        )

    @classmethod
    def approximate_particle(
        cls, diagnostics: Mapping[str, Any] | None = None
    ) -> "EvidenceSupport":
        """Monte Carlo or particle approximation to evidence."""

        return cls(
            support_type="approximate_particle",
            comparable=False,
            lower_bound=False,
            diagnostics={} if diagnostics is None else dict(diagnostics),
        )

    @classmethod
    def unknown(cls, diagnostics: Mapping[str, Any] | None = None) -> "EvidenceSupport":
        """Unknown or not-yet-declared evidence-support semantics."""

        return cls(
            support_type="unknown",
            comparable=False,
            lower_bound=False,
            diagnostics={} if diagnostics is None else dict(diagnostics),
        )


def coerce_evidence_support(value: Any) -> EvidenceSupport:
    """Coerce a mapping, string, or :class:`EvidenceSupport` into metadata."""

    if isinstance(value, EvidenceSupport):
        return value
    if isinstance(value, str):
        return EvidenceSupport(support_type=value)
    if isinstance(value, Mapping):
        return EvidenceSupport.from_mapping(value)
    return EvidenceSupport.unknown({"raw_value": repr(value)})


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
        known = {dataclass_field.name for dataclass_field in fields(cls)}
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
        normalized = _normalized_nonnegative_weights(values)
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
class ParticleFilterResult(_DiagnosticsMappingMixin):
    """Sequence-level particle-filter estimates and diagnostics.

    This container is intentionally generic: algorithms can store estimates,
    effective-sample-size histories, resampling decisions, spread summaries, and
    optional block-wise ESS values without tying the diagnostics module to a
    specific state space.
    """

    estimates: Any
    effective_sample_size: Any
    resampled: Any
    particle_spread: Any | None = None
    block_effective_sample_size: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ess_history(self) -> Any:
        return self.effective_sample_size

    @property
    def resampling_flags(self) -> Any:
        return self.resampled

    @property
    def resampling_count(self) -> int:
        return sum(1 for value in _coerce_bool_values(self.resampled) if value)

    @property
    def resampling_fraction(self) -> float:
        values = _coerce_bool_values(self.resampled)
        if not values:
            return 0.0
        return self.resampling_count / len(values)

    def summary_statistics(self) -> dict[str, Any]:
        """Return scalar sequence diagnostics for reports and logs."""
        ess = _coerce_numeric_values(self.effective_sample_size)
        spread = _coerce_numeric_values(self.particle_spread)
        block_ess = _coerce_numeric_values(self.block_effective_sample_size)

        summary = {
            "mean_effective_sample_size": _finite_mean(ess),
            "min_effective_sample_size": _finite_min(ess),
            "final_effective_sample_size": _finite_last(ess),
            "resampling_count": self.resampling_count,
            "resampling_fraction": self.resampling_fraction,
        }
        if spread:
            summary.update(
                {
                    "mean_particle_spread": _finite_mean(spread),
                    "final_particle_spread": _finite_last(spread),
                }
            )
        if block_ess:
            summary.update(
                {
                    "mean_block_effective_sample_size": _finite_mean(block_ess),
                    "min_block_effective_sample_size": _finite_min(block_ess),
                }
            )
        return summary


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
    "EvidenceSupport",
    "EvidenceSupportType",
    "FilterDiagnostics",
    "ParticleDiagnostics",
    "ParticleFilterResult",
    "coerce_evidence_support",
]
