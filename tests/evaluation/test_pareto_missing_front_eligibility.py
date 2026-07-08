from __future__ import annotations

import numpy as np
import pandas as pd
from pyrecest.evaluation import is_pareto_front, pareto_front_indices


def test_pareto_front_excludes_rows_without_comparable_objectives() -> None:
    table = pd.DataFrame(
        [
            {"name": "valid", "error": 1.0, "runtime": 1.0},
            {"name": "unknown", "error": np.nan, "runtime": np.nan},
        ]
    )

    indices = pareto_front_indices(
        table,
        ["error", "runtime"],
        directions={"error": "min", "runtime": "min"},
    )
    mask = is_pareto_front(
        table,
        ["error", "runtime"],
        directions={"error": "min", "runtime": "min"},
    )

    assert table.loc[indices, "name"].tolist() == ["valid"]
    assert mask.tolist() == [True, False]


def test_pareto_front_excludes_rows_with_only_nonnumeric_objectives() -> None:
    table = pd.DataFrame(
        [
            {"name": "valid", "error": 1.0, "runtime": 1.0},
            {"name": "nonnumeric", "error": "unavailable", "runtime": b"bad"},
        ]
    )

    indices = pareto_front_indices(
        table,
        ["error", "runtime"],
        directions={"error": "min", "runtime": "min"},
    )
    mask = is_pareto_front(
        table,
        ["error", "runtime"],
        directions={"error": "min", "runtime": "min"},
    )

    assert table.loc[indices, "name"].tolist() == ["valid"]
    assert mask.tolist() == [True, False]


def test_strict_pareto_front_requires_complete_objectives() -> None:
    table = pd.DataFrame(
        [
            {"name": "complete", "error": 1.0, "runtime": 1.0},
            {"name": "partial", "error": 0.5, "runtime": np.nan},
        ]
    )

    indices = pareto_front_indices(
        table,
        ["error", "runtime"],
        directions={"error": "min", "runtime": "min"},
        allow_missing=False,
    )
    mask = is_pareto_front(
        table,
        ["error", "runtime"],
        directions={"error": "min", "runtime": "min"},
        allow_missing=False,
    )

    assert table.loc[indices, "name"].tolist() == ["complete"]
    assert mask.tolist() == [True, False]
