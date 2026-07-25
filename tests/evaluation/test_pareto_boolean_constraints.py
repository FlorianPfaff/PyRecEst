from __future__ import annotations

import pandas as pd
import pytest
from pyrecest.evaluation import constraint_mask, select_under_constraints


def test_constraint_mask_accepts_boolean_thresholds() -> None:
    table = pd.DataFrame(
        [
            {"name": "enabled", "eligible": True},
            {"name": "disabled", "eligible": False},
            {"name": "missing", "eligible": None},
            {"name": "text", "eligible": "unknown"},
        ]
    )

    enabled = constraint_mask(table, {"eligible": ("==", True)})
    disabled = constraint_mask(table, {"eligible": {"op": "!=", "value": True}})

    assert table.loc[enabled, "name"].tolist() == ["enabled"]
    assert table.loc[disabled, "name"].tolist() == ["disabled"]


def test_constraint_mask_accepts_nullable_boolean_thresholds() -> None:
    table = pd.DataFrame(
        {
            "name": ["keep", "drop", "unknown"],
            "allowed": pd.Series([True, False, pd.NA], dtype="boolean"),
            "score": [2.0, 1.0, 0.0],
        }
    )

    true_mask = constraint_mask(table, {"allowed": ("==", True)})
    false_mask = constraint_mask(table, {"allowed": ("!=", True)})

    assert table.loc[true_mask, "name"].tolist() == ["keep"]
    assert table.loc[false_mask, "name"].tolist() == ["drop"]


def test_boolean_constraints_treat_non_boolean_cells_as_infeasible() -> None:
    table = pd.DataFrame(
        {
            "name": ["bool_true", "text_true", "numeric_one", "missing"],
            "allowed": [True, "True", 1, None],
        }
    )

    mask = constraint_mask(table, {"allowed": {"op": "==", "value": True}})

    assert table.loc[mask, "name"].tolist() == ["bool_true"]


def test_boolean_constraints_support_duplicate_index_labels() -> None:
    table = pd.DataFrame(
        {"allowed": [True, False, True]},
        index=["same", "same", "other"],
    )

    mask = constraint_mask(table, {"allowed": ("==", True)})

    assert mask.tolist() == [True, False, True]
    assert mask.index.tolist() == ["same", "same", "other"]


def test_boolean_constraints_work_through_selection() -> None:
    table = pd.DataFrame(
        {
            "name": ["allowed_slow", "blocked_fast", "allowed_fast"],
            "allowed": [True, False, True],
            "cost": [3.0, 0.5, 1.0],
        }
    )

    selected = select_under_constraints(
        table,
        constraints={"allowed": ("==", True)},
        objective="cost",
        direction="min",
    )

    assert selected["name"].tolist() == ["allowed_fast", "allowed_slow"]


def test_boolean_constraints_reject_ordering_operators() -> None:
    table = pd.DataFrame({"allowed": [True, False]})

    with pytest.raises(
        ValueError,
        match="Constraint threshold for 'allowed' must be a finite scalar",
    ):
        constraint_mask(table, {"allowed": ("<=", True)})
