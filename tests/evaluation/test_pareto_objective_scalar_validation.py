from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pyrecest.evaluation import (
    constraint_mask,
    is_pareto_front,
    pareto_front_indices,
    record_dominates,
)


def test_record_dominates_treats_numeric_text_objectives_as_missing() -> None:
    assert not record_dominates(
        {"objective": "0.0"},
        {"objective": 1.0},
        ["objective"],
        directions={"objective": "min"},
    )
    assert not record_dominates(
        {"objective": "0.0"},
        {"objective": 1.0},
        ["objective"],
        directions={"objective": "min"},
        allow_missing=False,
    )


def test_pareto_front_excludes_rows_with_only_numeric_text_objectives() -> None:
    table = pd.DataFrame(
        [
            {"name": "text_fast", "runtime": "0.0"},
            {"name": "numeric_slow", "runtime": 1.0},
        ]
    )

    indices = pareto_front_indices(table, ["runtime"], directions={"runtime": "min"})
    mask = is_pareto_front(table, ["runtime"], directions={"runtime": "min"})

    assert table.loc[indices, "name"].tolist() == ["numeric_slow"]
    assert mask.tolist() == [False, True]


def test_masked_objective_is_treated_as_missing() -> None:
    masked_fast = np.ma.array(0.0, mask=True)

    assert not record_dominates(
        {"objective": masked_fast},
        {"objective": 1.0},
        ["objective"],
        directions={"objective": "min"},
    )
    assert not record_dominates(
        {"objective": masked_fast},
        {"objective": 1.0},
        ["objective"],
        directions={"objective": "min"},
        allow_missing=False,
    )

    table = pd.DataFrame(
        [
            {"name": "masked_fast", "runtime": masked_fast},
            {"name": "numeric_slow", "runtime": 1.0},
        ]
    )
    indices = pareto_front_indices(table, ["runtime"], directions={"runtime": "min"})
    mask = is_pareto_front(table, ["runtime"], directions={"runtime": "min"})

    assert table.loc[indices, "name"].tolist() == ["numeric_slow"]
    assert mask.tolist() == [False, True]


def test_clear_mask_objective_remains_numeric() -> None:
    clear_mask_fast = np.ma.array(0.0, mask=False)

    assert record_dominates(
        {"objective": clear_mask_fast},
        {"objective": 1.0},
        ["objective"],
        directions={"objective": "min"},
    )


def test_masked_eps_is_rejected() -> None:
    with pytest.raises(ValueError, match="eps must be a finite non-negative scalar"):
        record_dominates(
            {"objective": 0.0},
            {"objective": 1.0},
            ["objective"],
            directions={"objective": "min"},
            eps=np.ma.array(0.0, mask=True),
        )


def test_masked_constraint_threshold_is_rejected() -> None:
    table = pd.DataFrame({"score": [1.0]})

    with pytest.raises(ValueError, match="Constraint threshold for 'score'"):
        constraint_mask(
            table,
            {"score": ("<=", np.ma.array(1.0, mask=True))},
        )

    assert constraint_mask(
        table,
        {"score": ("<=", np.ma.array(1.0, mask=False))},
    ).tolist() == [True]
