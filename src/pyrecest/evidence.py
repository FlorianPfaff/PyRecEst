"""Generic evidence-computation mode helpers.

Many filters and smoothers can compute the model evidence with a forward pass
without also constructing a fixed-interval smoothed posterior.  This module
provides a small domain-neutral way to make that choice explicit and to expose
consistent diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

EvidenceComputationKind = Literal["full_smoothing", "evidence_only"]


def _coerce_bool_flag(value: bool, name: str) -> bool:
    """Return a bool flag without treating arbitrary truthy objects as true."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{name} must be a bool")


@dataclass(frozen=True, slots=True)
class EvidenceComputationMode:
    """Declare whether an evidence computation should also smooth posteriors.

    Parameters
    ----------
    mode:
        ``"full_smoothing"`` requests the standard forward/backward output;
        ``"evidence_only"`` requests forward evidence and terminal posterior
        only.
    return_smoothed:
        Whether fixed-interval smoothed posterior marginals should be returned.
    terminal_posterior:
        Whether a terminal posterior is expected.  Most evidence-only filters
        can return this cheaply from the final filtering distribution.
    metadata:
        Optional caller-provided metadata copied into diagnostics.
    """

    mode: EvidenceComputationKind = "full_smoothing"
    return_smoothed: bool = True
    terminal_posterior: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in {"full_smoothing", "evidence_only"}:
            raise ValueError(f"unknown evidence computation mode {self.mode!r}")
        return_smoothed = _coerce_bool_flag(self.return_smoothed, "return_smoothed")
        terminal_posterior = _coerce_bool_flag(
            self.terminal_posterior, "terminal_posterior"
        )
        if self.mode == "evidence_only" and return_smoothed:
            raise ValueError("evidence_only mode cannot return smoothed posteriors")
        if self.mode == "full_smoothing" and not return_smoothed:
            raise ValueError("full_smoothing mode must return smoothed posteriors")
        object.__setattr__(self, "return_smoothed", return_smoothed)
        object.__setattr__(self, "terminal_posterior", terminal_posterior)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def full_smoothing(
        cls, *, metadata: dict[str, Any] | None = None
    ) -> "EvidenceComputationMode":
        """Return a mode that computes full fixed-interval smoothing."""

        return cls(
            mode="full_smoothing",
            return_smoothed=True,
            terminal_posterior=True,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def evidence_only(
        cls, *, metadata: dict[str, Any] | None = None
    ) -> "EvidenceComputationMode":
        """Return a mode that skips smoothing and keeps evidence semantics."""

        return cls(
            mode="evidence_only",
            return_smoothed=False,
            terminal_posterior=True,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def from_return_smoothed(cls, return_smoothed: bool) -> "EvidenceComputationMode":
        """Build a mode from the common Boolean smoothing flag."""

        parsed = _coerce_return_smoothed(return_smoothed)
        if parsed is None:
            raise ValueError("return_smoothed must be a bool")
        return cls.full_smoothing() if parsed else cls.evidence_only()

    @property
    def evidence_only_requested(self) -> bool:
        """Whether this mode is the evidence-only fast path."""

        return self.mode == "evidence_only"

    def to_diagnostics(self, prefix: str = "evidence") -> dict[str, Any]:
        """Return stable scalar diagnostics for reports and artifacts."""

        diagnostics = {
            f"{prefix}_computation_mode": self.mode,
            f"{prefix}_only": int(self.evidence_only_requested),
            f"{prefix}_return_smoothed": int(self.return_smoothed),
            f"{prefix}_terminal_posterior": int(self.terminal_posterior),
        }
        diagnostics.update(
            {f"{prefix}_{key}": value for key, value in self.metadata.items()}
        )
        return diagnostics


def _coerce_return_smoothed(return_smoothed: bool | None) -> bool | None:
    """Return an explicit Boolean smoothing flag without truthiness coercion."""

    if return_smoothed is None:
        return None
    if isinstance(return_smoothed, (bool, np.bool_)):
        return bool(return_smoothed)
    raise ValueError("return_smoothed must be a bool or None")


def _require_return_smoothed_agreement(
    mode: EvidenceComputationMode, return_smoothed: bool | None
) -> EvidenceComputationMode:
    """Reject contradictory explicit mode and compatibility flag requests."""

    return_smoothed = _coerce_return_smoothed(return_smoothed)
    if return_smoothed is None:
        return mode
    if return_smoothed != mode.return_smoothed:
        raise ValueError("mode and return_smoothed request inconsistent smoothing")
    return mode


def resolve_evidence_computation_mode(
    mode: EvidenceComputationMode | str | None = None,
    *,
    return_smoothed: bool | None = None,
) -> EvidenceComputationMode:
    """Resolve a string/Boolean compatibility mode into a typed object."""

    return_smoothed = _coerce_return_smoothed(return_smoothed)
    if isinstance(mode, EvidenceComputationMode):
        return _require_return_smoothed_agreement(mode, return_smoothed)
    if mode is None:
        return EvidenceComputationMode.from_return_smoothed(
            True if return_smoothed is None else return_smoothed
        )

    key = str(mode).strip().lower().replace("-", "_")
    if key in {"full", "full_smoothing", "smoothed", "smoothing"}:
        return _require_return_smoothed_agreement(
            EvidenceComputationMode.full_smoothing(), return_smoothed
        )
    if key in {
        "evidence",
        "evidence_only",
        "forward_only",
        "filter_only",
        "no_smoothing",
    }:
        return _require_return_smoothed_agreement(
            EvidenceComputationMode.evidence_only(), return_smoothed
        )
    raise ValueError(f"unknown evidence computation mode {mode!r}")


__all__ = [
    "EvidenceComputationKind",
    "EvidenceComputationMode",
    "resolve_evidence_computation_mode",
]
