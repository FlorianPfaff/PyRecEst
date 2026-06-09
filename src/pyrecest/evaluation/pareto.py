"""Generic Pareto-front and equal-quality selection utilities.

The helpers in this module intentionally stay domain-neutral.  Callers provide a
table, objective columns, objective directions, and optional quality constraints.
This makes the same machinery usable for filter/tracker benchmarks, compression
or pruning sweeps, particle-count trade-offs, and other rate--distortion style
selection problems.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

ObjectiveDirection = Literal["min", "max"]
ConstraintOperator = Literal["<=", ">=", "<", ">", "==", "!="]
ConstraintSpec = tuple[ConstraintOperator, float | int | bool]


def pareto_front_indices(
    table: pd.DataFrame,
    objectives: Sequence[str],
    *,
    directions: Mapping[str, ObjectiveDirection] | Sequence[ObjectiveDirection],
    feasible_mask: Sequence[bool] | pd.Series | np.ndarray | None = None,
    eps: float = 1e-12,
    allow_missing: bool = True,
) -> list[Any]:
    """Return indices of non-dominated rows.

    Parameters
    ----------
    table:
        DataFrame whose rows are candidates.
    objectives:
        Objective column names to compare.
    directions:
        Either a mapping from objective name to ``"min"``/``"max"`` or a
        sequence aligned with ``objectives``.
    feasible_mask:
        Optional row mask.  If supplied, only feasible rows can be on the front.
    eps:
        Numerical tolerance for weak/strict comparisons.
    allow_missing:
        If true, objective values that are missing in either row are skipped for
        that pair.  A row can dominate another only when at least one comparable
        objective remains and one comparable objective is strictly better.
    """

    if table.empty:
        return []
    objective_names = _validate_objectives(objectives)
    direction_map = _directions_by_objective(objective_names, directions)
    candidates = (
        table.loc[_feasible_index(table, feasible_mask)]
        if feasible_mask is not None
        else table
    )
    indices: list[Any] = []
    for index, row in candidates.iterrows():
        dominated = False
        for other_index, other in candidates.iterrows():
            if index == other_index:
                continue
            if record_dominates(
                other,
                row,
                objective_names,
                directions=direction_map,
                eps=eps,
                allow_missing=allow_missing,
            ):
                dominated = True
                break
        if not dominated:
            indices.append(index)
    return indices


def is_pareto_front(
    table: pd.DataFrame,
    objectives: Sequence[str],
    *,
    directions: Mapping[str, ObjectiveDirection] | Sequence[ObjectiveDirection],
    feasible_mask: Sequence[bool] | pd.Series | np.ndarray | None = None,
    eps: float = 1e-12,
    allow_missing: bool = True,
) -> pd.Series:
    """Return a boolean Series marking non-dominated rows."""

    mask = pd.Series(False, index=table.index, dtype=bool)
    mask.loc[
        pareto_front_indices(
            table,
            objectives,
            directions=directions,
            feasible_mask=feasible_mask,
            eps=eps,
            allow_missing=allow_missing,
        )
    ] = True
    return mask


def record_dominates(
    left: Mapping[str, Any] | pd.Series,
    right: Mapping[str, Any] | pd.Series,
    objectives: Sequence[str],
    *,
    directions: Mapping[str, ObjectiveDirection] | Sequence[ObjectiveDirection],
    eps: float = 1e-12,
    allow_missing: bool = True,
) -> bool:
    """Return whether ``left`` Pareto-dominates ``right``.

    ``left`` dominates ``right`` if it is at least as good on all comparable
    objectives and strictly better on at least one comparable objective.
    """

    objective_names = _validate_objectives(objectives)
    direction_map = _directions_by_objective(objective_names, directions)
    weak: list[bool] = []
    strict: list[bool] = []
    for objective in objective_names:
        left_value = _lookup_numeric(left, objective)
        right_value = _lookup_numeric(right, objective)
        if _is_missing(left_value) or _is_missing(right_value):
            if allow_missing:
                continue
            return False
        if direction_map[objective] == "min":
            weak.append(left_value <= right_value + eps)
            strict.append(left_value < right_value - eps)
        else:
            weak.append(left_value >= right_value - eps)
            strict.append(left_value > right_value + eps)
    return bool(weak) and all(weak) and any(strict)


def constraint_mask(
    table: pd.DataFrame,
    constraints: Mapping[str, ConstraintSpec | Mapping[str, Any]],
    *,
    eps: float = 1e-12,
) -> pd.Series:
    """Return rows satisfying all scalar constraints.

    Constraint values may be either ``("<=", value)`` tuples or mappings with
    ``{"op": "<=", "value": value}`` keys.
    """

    mask = pd.Series(True, index=table.index, dtype=bool)
    for column, spec in constraints.items():
        op, threshold = _parse_constraint_spec(spec)
        if column not in table.columns:
            mask &= False
            continue
        values = pd.to_numeric(table[column], errors="coerce")
        if op == "<=":
            mask &= values <= float(threshold) + eps
        elif op == ">=":
            mask &= values >= float(threshold) - eps
        elif op == "<":
            mask &= values < float(threshold) - eps
        elif op == ">":
            mask &= values > float(threshold) + eps
        elif op == "==":
            mask &= np.isclose(values, float(threshold), atol=eps, rtol=0.0)
        elif op == "!=":
            mask &= ~np.isclose(values, float(threshold), atol=eps, rtol=0.0)
        else:  # pragma: no cover - protected by _parse_constraint_spec.
            raise ValueError(f"Unsupported constraint operator {op!r}.")
    return mask.fillna(False)


def select_under_constraints(
    table: pd.DataFrame,
    *,
    constraints: Mapping[str, ConstraintSpec | Mapping[str, Any]],
    objective: str,
    direction: ObjectiveDirection,
    tie_breakers: Sequence[tuple[str, ObjectiveDirection]] = (),
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Return feasible rows sorted by one objective plus optional tie-breakers."""

    if objective not in table.columns:
        raise ValueError(f"objective column {objective!r} is not present.")
    sorted_columns = [objective, *(column for column, _ in tie_breakers)]
    missing = [column for column in sorted_columns if column not in table.columns]
    if missing:
        raise ValueError(f"selection columns are missing: {', '.join(missing)}.")
    feasible = table.loc[constraint_mask(table, constraints, eps=eps)].copy()
    if feasible.empty:
        return feasible
    ascending = [
        direction == "min",
        *(tie_direction == "min" for _, tie_direction in tie_breakers),
    ]
    return feasible.sort_values(sorted_columns, ascending=ascending, na_position="last")


def equal_quality_selection(
    table: pd.DataFrame,
    *,
    quality_constraints: Mapping[str, ConstraintSpec | Mapping[str, Any]],
    compression_objective: str,
    compression_direction: ObjectiveDirection = "min",
    tie_breakers: Sequence[tuple[str, ObjectiveDirection]] = (),
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Select candidates under fixed quality constraints.

    This is a named wrapper around :func:`select_under_constraints` for common
    equal-quality compression/compaction comparisons.
    """

    return select_under_constraints(
        table,
        constraints=quality_constraints,
        objective=compression_objective,
        direction=compression_direction,
        tie_breakers=tie_breakers,
        eps=eps,
    )


def _validate_objectives(objectives: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(objective) for objective in objectives)
    if not names:
        raise ValueError("At least one Pareto objective is required.")
    if len(set(names)) != len(names):
        raise ValueError("Pareto objectives must be unique.")
    return names


def _directions_by_objective(
    objectives: Sequence[str],
    directions: Mapping[str, ObjectiveDirection] | Sequence[ObjectiveDirection],
) -> dict[str, ObjectiveDirection]:
    if isinstance(directions, Mapping):
        missing = [objective for objective in objectives if objective not in directions]
        if missing:
            raise ValueError(f"Missing objective directions for: {', '.join(missing)}.")
        result = {objective: directions[objective] for objective in objectives}
    else:
        if len(directions) != len(objectives):
            raise ValueError("directions must match the number of objectives.")
        result = dict(zip(objectives, directions, strict=True))
    invalid = [
        objective
        for objective, direction in result.items()
        if direction not in {"min", "max"}
    ]
    if invalid:
        raise ValueError(f"Invalid objective directions for: {', '.join(invalid)}.")
    return result


def _feasible_index(
    table: pd.DataFrame,
    feasible_mask: Sequence[bool] | pd.Series | np.ndarray,
) -> pd.Series:
    if isinstance(feasible_mask, pd.Series):
        mask = feasible_mask.reindex(table.index, fill_value=False)
    else:
        mask = pd.Series(feasible_mask, index=table.index)
    return mask.astype(bool)


def _lookup_numeric(record: Mapping[str, Any] | pd.Series, key: str) -> float:
    value = record.get(key, np.nan)
    if _is_missing(value):
        return float("nan")
    return float(value)


def _is_missing(value: Any) -> bool:
    return bool(pd.isna(value))


def _parse_constraint_spec(spec: ConstraintSpec | Mapping[str, Any]) -> ConstraintSpec:
    if isinstance(spec, Mapping):
        op = spec.get("op")
        threshold = spec.get("value")
    else:
        if len(spec) != 2:
            raise ValueError("Constraint tuple specs must have length 2.")
        op, threshold = spec
    if op not in {"<=", ">=", "<", ">", "==", "!="}:
        raise ValueError(f"Unsupported constraint operator {op!r}.")
    return op, threshold
