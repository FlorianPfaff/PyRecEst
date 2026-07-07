"""Composable cost-matrix adjustment helpers.

These helpers provide a small domain-neutral interface for applying priors,
uncertainty corrections, consistency penalties, or other transformations to
pairwise association cost matrices before assignment.  Domain-specific projects
should construct the matrices and any metadata themselves, then expose only a
matrix-shaped adjustment to PyRecEst.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Protocol

import numpy as np


def _metadata_dict(value: Any, *, name: str) -> dict[str, Any]:
    """Return a diagnostics/metadata dictionary, treating ``None`` as absent."""

    if value is None:
        return {}
    try:
        return dict(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a mapping or None") from exc


@dataclass(frozen=True)
class CostMatrixAdjustmentResult:
    """Result of applying one or more shape-preserving cost adjustments."""

    adjusted_cost_matrix: np.ndarray
    diagnostics: Mapping[str, Any] | None = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "diagnostics",
            _metadata_dict(self.diagnostics, name="diagnostics"),
        )


class CostMatrixAdjustment(Protocol):
    """Protocol for domain-neutral cost-matrix adjustments."""

    name: str

    def apply(
        self,
        cost_matrix: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> CostMatrixAdjustmentResult | Any:
        """Return an adjusted matrix or a ``CostMatrixAdjustmentResult``."""


AdjustmentFunction = Callable[[np.ndarray, Mapping[str, Any]], Any]


@dataclass(frozen=True)
class CallableCostMatrixAdjustment:
    """Cost adjustment backed by a callable.

    The callable receives a validated numeric matrix and merged metadata.  It may
    return either an adjusted matrix, a ``(matrix, diagnostics)`` tuple, or a
    ``CostMatrixAdjustmentResult``.
    """

    name: str
    function: AdjustmentFunction
    metadata: Mapping[str, Any] | None = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if not callable(self.function):
            raise ValueError("function must be callable")
        object.__setattr__(
            self,
            "metadata",
            _metadata_dict(self.metadata, name="metadata"),
        )

    def apply(
        self,
        cost_matrix: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> CostMatrixAdjustmentResult:
        """Apply the wrapped callable to ``cost_matrix``."""

        matrix = _as_cost_matrix(cost_matrix)
        merged_metadata = dict(self.metadata)
        merged_metadata.update(_metadata_dict(metadata, name="metadata"))
        return _coerce_adjustment_output(
            self.function(matrix.copy(), merged_metadata),
            expected_shape=matrix.shape,
        )


def apply_cost_matrix_adjustment(
    cost_matrix: Any,
    adjustment: (
        CostMatrixAdjustment
        | Callable[[np.ndarray], Any]
        | CallableCostMatrixAdjustment
    ),
    *,
    metadata: Mapping[str, Any] | None = None,
) -> CostMatrixAdjustmentResult:
    """Apply one shape-preserving adjustment to a cost matrix.

    ``adjustment`` may be an object exposing ``apply(cost_matrix, metadata=...)``
    or a simple callable that accepts a validated matrix.  The adjusted matrix is
    validated to have the same shape as the input.
    """

    matrix = _as_cost_matrix(cost_matrix)
    if hasattr(adjustment, "apply"):
        output = adjustment.apply(matrix.copy(), metadata=metadata)  # type: ignore[attr-defined]
    elif callable(adjustment):
        output = adjustment(matrix.copy())
    else:
        raise ValueError("adjustment must be callable or expose an apply method")
    return _coerce_adjustment_output(output, expected_shape=matrix.shape)


def compose_cost_matrix_adjustments(
    cost_matrix: Any,
    adjustments: Sequence[
        CostMatrixAdjustment
        | Callable[[np.ndarray], Any]
        | CallableCostMatrixAdjustment
    ],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> CostMatrixAdjustmentResult:
    """Apply cost-matrix adjustments in order and collect diagnostics."""

    matrix = _as_cost_matrix(cost_matrix)
    diagnostics: dict[str, Any] = {"adjustment_order": []}
    diagnostic_keys: set[str] = set()
    current = matrix
    for index, adjustment in enumerate(adjustments):
        base_name = _adjustment_name(adjustment, index=index)
        name = base_name
        suffix = 2
        while name in diagnostic_keys:
            name = f"{base_name}_{suffix}"
            suffix += 1
        diagnostic_keys.add(name)
        result = apply_cost_matrix_adjustment(
            current,
            adjustment,
            metadata=metadata,
        )
        current = result.adjusted_cost_matrix
        diagnostics["adjustment_order"].append(name)
        diagnostics[name] = dict(result.diagnostics)
    return CostMatrixAdjustmentResult(current, diagnostics)


def additive_cost_matrix_adjustment(
    penalty_matrix: Any,
    *,
    name: str = "additive_penalty",
    diagnostics: Mapping[str, Any] | None = None,
) -> CallableCostMatrixAdjustment:
    """Return an adjustment that adds a same-shaped penalty matrix."""

    penalty = _as_cost_matrix(penalty_matrix)
    stored_diagnostics = _metadata_dict(diagnostics, name="diagnostics")

    def _add(
        matrix: np.ndarray,
        _metadata: Mapping[str, Any],
    ) -> CostMatrixAdjustmentResult:
        if matrix.shape != penalty.shape:
            raise ValueError(
                f"penalty_matrix shape {penalty.shape} does not match cost_matrix shape {matrix.shape}"
            )
        return CostMatrixAdjustmentResult(matrix + penalty, stored_diagnostics)

    return CallableCostMatrixAdjustment(name=name, function=_add)


def _adjustment_name(adjustment: Any, *, index: int) -> str:
    name = getattr(adjustment, "name", None)
    if isinstance(name, str) and name:
        return name
    callable_name = getattr(adjustment, "__name__", None)
    if isinstance(callable_name, str) and callable_name:
        return callable_name
    return f"adjustment_{index}"


def _coerce_adjustment_output(
    output: Any,
    *,
    expected_shape: tuple[int, int],
) -> CostMatrixAdjustmentResult:
    if isinstance(output, CostMatrixAdjustmentResult):
        adjusted = _as_cost_matrix(output.adjusted_cost_matrix)
        diagnostics = _metadata_dict(output.diagnostics, name="diagnostics")
    elif isinstance(output, tuple) and len(output) == 2:
        adjusted = _as_cost_matrix(output[0])
        diagnostics = _metadata_dict(output[1], name="diagnostics")
    else:
        adjusted = _as_cost_matrix(output)
        diagnostics = {}
    if adjusted.shape != expected_shape:
        raise ValueError(
            f"adjusted cost matrix has shape {adjusted.shape}, expected {expected_shape}"
        )
    return CostMatrixAdjustmentResult(adjusted, diagnostics)


def _as_cost_matrix(value: Any) -> np.ndarray:
    if _contains_invalid_values(value):
        raise ValueError("cost_matrix must be real-valued numeric")
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("cost_matrix must be real-valued numeric") from exc
    if matrix.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional")
    if matrix.dtype == np.bool_:
        raise ValueError("cost_matrix must be real-valued numeric")
    if np.any(np.isneginf(matrix)):
        raise ValueError("cost_matrix may not contain negative infinity")
    return np.nan_to_num(matrix, nan=np.inf, posinf=np.inf)


def _contains_invalid_values(value: Any) -> bool:
    try:
        flat = np.asarray(value, dtype=object).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return True
    for item in flat:
        if item is None or isinstance(item, (bool, np.bool_, str, bytes, bytearray)):
            return True
        if isinstance(item, (complex, np.complexfloating)):
            return True
        if not isinstance(item, Real) and not np.isscalar(item):
            return True
    return False


__all__ = (
    "CallableCostMatrixAdjustment",
    "CostMatrixAdjustment",
    "CostMatrixAdjustmentResult",
    "additive_cost_matrix_adjustment",
    "apply_cost_matrix_adjustment",
    "compose_cost_matrix_adjustments",
)
