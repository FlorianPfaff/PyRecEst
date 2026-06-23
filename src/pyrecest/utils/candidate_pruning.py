"""Candidate pruning helpers for pairwise association matrices.

These utilities reduce dense pairwise association matrices to high-recall
candidate graphs before downstream assignment.  Pruning is expressed as a boolean
mask over a rectangular cost matrix; callers that need to preserve matrix shape
can replace pruned entries by a large finite cost.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CandidatePruningConfig:
    """Configuration for pairwise cost/probability candidate pruning.

    The enabled criteria are combined by union: a candidate is kept when any
    active row/column top-k, probability-threshold, cost-threshold, or percentile
    criterion selects it.  If no criterion is enabled, all finite costs are kept.
    """

    row_top_k: int | None = None
    column_top_k: int | None = None
    probability_threshold: float | None = None
    max_cost: float | None = None
    max_cost_percentile: float | None = None
    always_keep_finite: bool = False
    large_cost: float = 1.0e6

    def __post_init__(self) -> None:
        for name in ("row_top_k", "column_top_k"):
            value = getattr(self, name)
            if value is None:
                continue
            parsed = _normalize_positive_integer(value, name)
            object.__setattr__(self, name, parsed)

        if self.probability_threshold is not None:
            threshold = _normalize_bounded_scalar(
                self.probability_threshold,
                lower=0.0,
                upper=1.0,
                message="probability_threshold must lie in [0, 1]",
            )
            object.__setattr__(self, "probability_threshold", threshold)

        if self.max_cost is not None:
            max_cost = _normalize_finite_scalar(
                self.max_cost,
                message="max_cost must be finite or None",
            )
            object.__setattr__(self, "max_cost", max_cost)

        if self.max_cost_percentile is not None:
            percentile = _normalize_bounded_scalar(
                self.max_cost_percentile,
                lower=0.0,
                upper=100.0,
                message="max_cost_percentile must lie in [0, 100]",
            )
            object.__setattr__(self, "max_cost_percentile", percentile)

        large_cost = _normalize_finite_scalar(
            self.large_cost,
            message="large_cost must be finite and positive",
        )
        if large_cost <= 0.0:
            raise ValueError("large_cost must be finite and positive")
        object.__setattr__(self, "large_cost", large_cost)


def candidate_pruning_config_from_mapping(
    value: CandidatePruningConfig | Mapping[str, Any] | None,
) -> CandidatePruningConfig | None:
    """Normalize optional pruning config inputs."""

    if value is None:
        return None
    if isinstance(value, CandidatePruningConfig):
        return value
    return CandidatePruningConfig(**dict(value))


def _normalize_positive_integer(value: Any, name: str) -> int:
    value_array = np.asarray(value)
    message = f"{name} must be a positive integer or None"
    if value_array.shape != () or value_array.dtype == np.bool_:
        raise ValueError(message)

    scalar = value_array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(scalar, (int, np.integer)):
        parsed = int(scalar)
    else:
        try:
            scalar_float = float(scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(message) from exc
        if not np.isfinite(scalar_float) or not scalar_float.is_integer():
            raise ValueError(message)
        parsed = int(scalar_float)

    if parsed <= 0:
        raise ValueError(message)
    return parsed


def _normalize_finite_scalar(value: Any, *, message: str) -> float:
    value_array = np.asarray(value)
    if value_array.shape != () or value_array.dtype == np.bool_:
        raise ValueError(message)

    scalar = value_array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(message)
    try:
        parsed = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(parsed):
        raise ValueError(message)
    return parsed


def _normalize_bounded_scalar(
    value: Any,
    *,
    lower: float,
    upper: float,
    message: str,
) -> float:
    parsed = _normalize_finite_scalar(value, message=message)
    if not lower <= parsed <= upper:
        raise ValueError(message)
    return parsed


def candidate_mask_from_costs(
    cost_matrix: Any,
    *,
    probability_matrix: Any | None = None,
    config: CandidatePruningConfig | Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Return a boolean candidate mask for a pairwise association matrix."""

    costs = _as_cost_matrix(cost_matrix)
    finite_costs = np.isfinite(costs)
    cfg = candidate_pruning_config_from_mapping(config)
    if cfg is None:
        return finite_costs

    keep = np.zeros(costs.shape, dtype=bool)
    any_rule = False

    if cfg.always_keep_finite:
        keep |= finite_costs
        any_rule = True
    if cfg.row_top_k is not None:
        keep |= _row_top_k_mask(costs, cfg.row_top_k)
        any_rule = True
    if cfg.column_top_k is not None:
        keep |= _column_top_k_mask(costs, cfg.column_top_k)
        any_rule = True
    if cfg.probability_threshold is not None:
        if probability_matrix is None:
            raise ValueError(
                "probability_matrix is required when probability_threshold is set"
            )
        probabilities = _as_probability_matrix(probability_matrix, costs.shape)
        keep |= np.isfinite(probabilities) & (
            probabilities >= cfg.probability_threshold
        )
        any_rule = True
    if cfg.max_cost is not None:
        keep |= finite_costs & (costs <= cfg.max_cost)
        any_rule = True
    if cfg.max_cost_percentile is not None:
        threshold = _finite_percentile(costs, cfg.max_cost_percentile)
        if threshold is not None:
            keep |= finite_costs & (costs <= threshold)
            any_rule = True

    if not any_rule:
        keep = finite_costs
    return keep & finite_costs


def prune_pairwise_cost_matrix(
    cost_matrix: Any,
    *,
    probability_matrix: Any | None = None,
    config: CandidatePruningConfig | Mapping[str, Any] | None = None,
    large_cost: float | None = None,
) -> np.ndarray:
    """Replace pruned candidate entries by a large finite cost."""

    costs = _as_cost_matrix(cost_matrix)
    cfg = candidate_pruning_config_from_mapping(config)
    if cfg is None:
        return costs

    if large_cost is None:
        penalty = cfg.large_cost
    else:
        penalty = _normalize_finite_scalar(
            large_cost,
            message="large_cost must be finite and positive",
        )
    if penalty <= 0.0:
        raise ValueError("large_cost must be finite and positive")
    penalty = _effective_large_cost(costs, penalty)

    mask = candidate_mask_from_costs(
        costs,
        probability_matrix=probability_matrix,
        config=cfg,
    )
    return np.where(mask, costs, penalty)


def _row_top_k_mask(costs: np.ndarray, top_k: int) -> np.ndarray:
    mask = np.zeros(costs.shape, dtype=bool)
    if costs.shape[1] == 0:
        return mask

    k = min(int(top_k), costs.shape[1])
    for row_index, row in enumerate(costs):
        finite_columns = np.flatnonzero(np.isfinite(row))
        if finite_columns.size == 0:
            continue
        ordered = finite_columns[np.argsort(row[finite_columns], kind="stable")]
        mask[row_index, ordered[:k]] = True
    return mask


def _column_top_k_mask(costs: np.ndarray, top_k: int) -> np.ndarray:
    mask = np.zeros(costs.shape, dtype=bool)
    if costs.shape[0] == 0:
        return mask

    k = min(int(top_k), costs.shape[0])
    for column_index in range(costs.shape[1]):
        column = costs[:, column_index]
        finite_rows = np.flatnonzero(np.isfinite(column))
        if finite_rows.size == 0:
            continue
        ordered = finite_rows[np.argsort(column[finite_rows], kind="stable")]
        mask[ordered[:k], column_index] = True
    return mask


def _finite_percentile(costs: np.ndarray, percentile: float) -> float | None:
    finite = np.asarray(costs, dtype=float)[np.isfinite(costs)]
    if finite.size == 0:
        return None
    return float(np.percentile(finite, float(percentile)))


def _effective_large_cost(costs: np.ndarray, penalty: float) -> float:
    finite = costs[np.isfinite(costs)]
    if finite.size == 0:
        return penalty

    max_finite = float(np.max(finite))
    if penalty > max_finite:
        return penalty

    adjusted_penalty = float(np.nextafter(max_finite, np.inf))
    if not np.isfinite(adjusted_penalty):
        raise ValueError("large_cost is too small to exceed finite costs")
    return adjusted_penalty


def _as_cost_matrix(cost_matrix: Any) -> np.ndarray:
    costs = np.asarray(cost_matrix, dtype=float)
    if costs.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional")
    return np.nan_to_num(costs, nan=np.inf, posinf=np.inf, neginf=np.inf)


def _as_probability_matrix(
    probability_matrix: Any,
    shape: tuple[int, int],
) -> np.ndarray:
    probabilities = np.asarray(probability_matrix, dtype=float)
    if probabilities.shape != shape:
        raise ValueError("probability_matrix must match cost_matrix shape")
    finite = np.isfinite(probabilities)
    if np.any(finite & ((probabilities < 0.0) | (probabilities > 1.0))):
        raise ValueError("finite probability_matrix entries must lie in [0, 1]")
    return probabilities


__all__ = (
    "CandidatePruningConfig",
    "candidate_mask_from_costs",
    "candidate_pruning_config_from_mapping",
    "prune_pairwise_cost_matrix",
)
