from __future__ import annotations

import pandas as pd
import pytest
from pyrecest.evaluation import (
    constraint_mask,
    equal_quality_selection,
    is_pareto_front,
    pareto_front_indices,
    record_dominates,
    select_under_constraints,
)


def test_pareto_front_indices_handles_min_and_max_objectives() -> None:
    table = pd.DataFrame(
        [
            {"name": "large_good", "size": 10.0, "quality": 0.95},
            {"name": "small_good", "size": 5.0, "quality": 0.95},
            {"name": "small_bad", "size": 5.0, "quality": 0.90},
            {"name": "tiny_ok", "size": 3.0, "quality": 0.91},
        ]
    )

    indices = pareto_front_indices(
        table, ["size", "quality"], directions={"size": "min", "quality": "max"}
    )

    assert set(table.loc[indices, "name"]) == {"small_good", "tiny_ok"}


def test_pareto_front_handles_duplicate_index_labels() -> None:
    table = pd.DataFrame(
        {"error": [2.0, 1.0], "runtime": [1.0, 1.0]},
        index=["same", "same"],
    )

    indices = pareto_front_indices(
        table, ["error", "runtime"], directions={"error": "min", "runtime": "min"}
    )
    mask = is_pareto_front(
        table, ["error", "runtime"], directions={"error": "min", "runtime": "min"}
    )

    assert indices == ["same"]
    assert mask.tolist() == [False, True]
    assert mask.index.tolist() == ["same", "same"]


def test_feasible_mask_treats_missing_nullable_values_as_infeasible() -> None:
    table = pd.DataFrame(
        [
            {"name": "balanced", "error": 1.0, "runtime": 2.0},
            {"name": "best_but_unknown", "error": 0.5, "runtime": 0.5},
            {"name": "fast", "error": 2.0, "runtime": 1.0},
        ]
    )
    feasible_mask = pd.Series([True, pd.NA, True], dtype="boolean")

    indices = pareto_front_indices(
        table,
        ["error", "runtime"],
        directions={"error": "min", "runtime": "min"},
        feasible_mask=feasible_mask,
    )
    mask = is_pareto_front(
        table,
        ["error", "runtime"],
        directions={"error": "min", "runtime": "min"},
        feasible_mask=feasible_mask,
    )

    assert indices == [0, 2]
    assert mask.tolist() == [True, False, True]


def test_is_pareto_front_respects_feasible_mask() -> None:
    table = pd.DataFrame(
        [
            {"name": "baseline", "size": 10.0, "quality": 1.0, "allowed": True},
            {"name": "fast", "size": 4.0, "quality": 0.9, "allowed": True},
            {"name": "forbidden", "size": 3.0, "quality": 0.99, "allowed": False},
        ]
    )

    mask = is_pareto_front(
        table,
        ["size", "quality"],
        directions=["min", "max"],
        feasible_mask=table["allowed"],
    )

    assert set(table.loc[mask, "name"]) == {"baseline", "fast"}


def test_record_dominates_can_skip_missing_optional_metrics() -> None:
    left = {"size": 4.0, "quality": 0.95, "optional": float("nan")}
    right = {"size": 5.0, "quality": 0.95, "optional": 2.0}

    assert record_dominates(
        left,
        right,
        ["size", "quality", "optional"],
        directions={"size": "min", "quality": "max", "optional": "max"},
    )
    assert not record_dominates(
        left,
        right,
        ["size", "quality", "optional"],
        directions={"size": "min", "quality": "max", "optional": "max"},
        allow_missing=False,
    )


def test_record_dominates_treats_non_numeric_metrics_as_missing_when_allowed() -> None:
    left = {"size": 4.0, "quality": 0.95, "optional": "not available"}
    right = {"size": 5.0, "quality": 0.95, "optional": 2.0}

    assert record_dominates(
        left,
        right,
        ["size", "quality", "optional"],
        directions={"size": "min", "quality": "max", "optional": "max"},
    )
    assert not record_dominates(
        left,
        right,
        ["size", "quality", "optional"],
        directions={"size": "min", "quality": "max", "optional": "max"},
        allow_missing=False,
    )


def test_record_dominates_treats_nonscalar_metrics_as_missing_when_allowed() -> None:
    left = {"size": 4.0, "quality": 0.95, "optional": [1.0, 2.0]}
    right = {"size": 5.0, "quality": 0.95, "optional": 2.0}

    assert record_dominates(
        left,
        right,
        ["size", "quality", "optional"],
        directions={"size": "min", "quality": "max", "optional": "max"},
    )
    assert not record_dominates(
        left,
        right,
        ["size", "quality", "optional"],
        directions={"size": "min", "quality": "max", "optional": "max"},
        allow_missing=False,
    )


def test_constraint_mask_supports_tuple_and_mapping_specs() -> None:
    table = pd.DataFrame(
        [
            {"name": "a", "drop": -0.02, "lpips": 0.002},
            {"name": "b", "drop": -0.08, "lpips": 0.002},
            {"name": "c", "drop": -0.01, "lpips": 0.006},
        ]
    )

    mask = constraint_mask(
        table, {"drop": (">=", -0.06), "lpips": {"op": "<=", "value": 0.0035}}
    )

    assert table.loc[mask, "name"].tolist() == ["a"]


def test_constraint_mask_treats_missing_not_equal_values_as_infeasible() -> None:
    table = pd.DataFrame(
        [
            {"name": "different", "score": 0.2},
            {"name": "same", "score": 0.0},
            {"name": "missing", "score": None},
            {"name": "non_numeric", "score": "unknown"},
        ]
    )

    mask = constraint_mask(table, {"score": ("!=", 0.0)})

    assert table.loc[mask, "name"].tolist() == ["different"]


def test_select_under_constraints_rejects_invalid_directions() -> None:
    table = pd.DataFrame([{"name": "a", "score": 1.0, "tie": 0.0}])

    with pytest.raises(ValueError, match="objective"):
        select_under_constraints(
            table,
            constraints={},
            objective="score",
            direction="sideways",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="tie-breaker"):
        select_under_constraints(
            table,
            constraints={},
            objective="score",
            direction="min",
            tie_breakers=(("tie", "sideways"),),  # type: ignore[arg-type]
        )


def test_equal_quality_selection_returns_best_compression_row() -> None:
    table = pd.DataFrame(
        [
            {
                "name": "safe_large",
                "retention": 0.8,
                "psnr_delta": -0.01,
                "lpips_delta": 0.001,
                "f_score": 0.24,
            },
            {
                "name": "safe_small",
                "retention": 0.5,
                "psnr_delta": -0.05,
                "lpips_delta": 0.003,
                "f_score": 0.20,
            },
            {
                "name": "unsafe_tiny",
                "retention": 0.4,
                "psnr_delta": -0.09,
                "lpips_delta": 0.003,
                "f_score": 0.26,
            },
        ]
    )

    selected = equal_quality_selection(
        table,
        quality_constraints={
            "psnr_delta": (">=", -0.06),
            "lpips_delta": ("<=", 0.0035),
        },
        compression_objective="retention",
        compression_direction="min",
        tie_breakers=(("f_score", "max"),),
    )

    assert selected.iloc[0]["name"] == "safe_small"
