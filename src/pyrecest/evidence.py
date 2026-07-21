"""Generic evidence-computation mode helpers.

Many filters and smoothers can compute the model evidence with a forward pass
without also constructing a fixed-interval smoothed posterior.  This module
provides a small domain-neutral way to make that choice explicit and to expose
consistent diagnostics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

EvidenceComputationKind = Literal["full_smoothing", "evidence_only"]
_RESERVED_DIAGNOSTIC_METADATA_KEYS = frozenset(
    {
        "computation_mode",
        "only",
        "return_smoothed",
        "terminal_posterior",
    }
)


def _coerce_bool_flag(value: bool, name: str) -> bool:
    """Return a bool flag without treating arbitrary truthy objects as true."""

    try:
        value_array = np.asarray(value)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a bool") from exc
    if value_array.shape == () and np.issubdtype(value_array.dtype, np.bool_):
        return bool(value_array.item())
    raise ValueError(f"{name} must be a bool")


def _coerce_metadata(metadata: Any) -> dict[str, Any]:
    """Return a plain metadata dictionary without sequence-to-dict surprises."""

    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping or None")
    return dict(metadata)


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
    metadata: dict[str, Any] | None = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in {
            "full_smoothing",
            "evidence_only",
        }:
            raise ValueError(f"unknown evidence computation mode {self.mode!r}")
        return_smoothed = _coerce_bool_flag(self.return_smoothed, "return_smoothed")
        terminal_posterior = _coerce_bool_flag(
            self.terminal_posterior, "terminal_posterior"
        )
        if self.mode == "evidence_only" and return_smoothed:
            raise ValueError("evidence_only mode cannot return smoothed posteriors")
        if self.mode == "full_smoothing" and not return_smoothed:
            raise ValueError("full_smoothing mode must return smoothed posteriors")
        metadata = _coerce_metadata(self.metadata)
        conflicting_metadata_keys = _RESERVED_DIAGNOSTIC_METADATA_KEYS.intersection(
            metadata
        )
        if conflicting_metadata_keys:
            reserved = ", ".join(sorted(str(key) for key in conflicting_metadata_keys))
            raise ValueError(
                "metadata keys would overwrite evidence diagnostics: " f"{reserved}"
            )
        object.__setattr__(self, "return_smoothed", return_smoothed)
        object.__setattr__(self, "terminal_posterior", terminal_posterior)
        object.__setattr__(self, "metadata", metadata)

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
    return _coerce_bool_flag(return_smoothed, "return_smoothed")


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
    if not isinstance(mode, str):
        raise ValueError(f"unknown evidence computation mode {mode!r}")

    key = mode.strip().lower().replace("-", "_")
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


try:
    from pyrecest._backend_runtime_patches import (
        patch_pytorch_close_equal_nan_device_contract as _patch_pytorch_close_equal_nan_device_contract,  # pylint: disable=import-outside-toplevel
    )
    from pyrecest._backend_runtime_patches import (
        patch_pytorch_edge_pad_contract as _patch_pytorch_edge_pad_contract,
    )
    from pyrecest._backend_runtime_patches import (
        patch_pytorch_repeat_numpy_contract as _patch_pytorch_repeat_numpy_contract,
    )
    from pyrecest._backend_runtime_patches import (
        patch_pytorch_searchsorted_contract as _patch_pytorch_searchsorted_contract,
    )
    from pyrecest._backend_runtime_patches import (
        patch_pytorch_take_axis_contract as _patch_pytorch_take_axis_contract,
    )
    from pyrecest._backend_runtime_patches import (
        patch_pytorch_transpose_boolean_axes_contract as _patch_pytorch_transpose_boolean_axes_contract,
    )
    from pyrecest._backend_runtime_patches import (
        patch_raw_pytorch_reduction_alias_contract as _patch_raw_pytorch_reduction_alias_contract,
    )
    from pyrecest.backend_support._jax_array_from_sparse_contract import (
        patch_jax_array_from_sparse_flat_index_contract as _patch_jax_array_from_sparse_flat_index_contract,  # pylint: disable=import-outside-toplevel
    )
    from pyrecest.backend_support._pytorch_nonzero_scalar_contract import (
        patch_pytorch_nonzero_scalar_contract as _patch_pytorch_nonzero_scalar_contract,  # pylint: disable=import-outside-toplevel
    )
    from pyrecest.backend_support._pytorch_one_hot_scalar_contract import (
        patch_pytorch_one_hot_scalar_contract as _patch_pytorch_one_hot_scalar_contract,  # pylint: disable=import-outside-toplevel
    )
    from pyrecest.backend_support._pytorch_quantile_empty_batch_contract import (
        patch_pytorch_quantile_empty_batch_contract as _patch_pytorch_quantile_empty_batch_contract,  # pylint: disable=import-outside-toplevel
    )
    from pyrecest.backend_support._pytorch_reduction_axis_contract import (
        patch_pytorch_reduction_axis_contract as _patch_pytorch_reduction_axis_contract,  # pylint: disable=import-outside-toplevel
    )
except ModuleNotFoundError:  # pragma: no cover - source tree corruption only
    pass
else:
    _patch_pytorch_close_equal_nan_device_contract()
    _patch_pytorch_repeat_numpy_contract()
    _patch_pytorch_edge_pad_contract()
    _patch_pytorch_searchsorted_contract()
    _patch_raw_pytorch_reduction_alias_contract()
    _patch_pytorch_take_axis_contract()
    _patch_pytorch_transpose_boolean_axes_contract()
    _patch_pytorch_nonzero_scalar_contract()
    _patch_pytorch_one_hot_scalar_contract()
    _patch_pytorch_quantile_empty_batch_contract()
    _patch_pytorch_reduction_axis_contract()
    _patch_jax_array_from_sparse_flat_index_contract()
