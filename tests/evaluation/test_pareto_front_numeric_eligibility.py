from __future__ import annotations

import pandas as pd

from pyrecest.evaluation import is_pareto_front, pareto_front_indices


def test_pareto_front_excludes_rows_without_numeric_objectives_when_missing_allowed() -> None:
    table = pd.DataFrame(
        [
            {"name": "invalid", "error": "unknown", "runtime": [1.0, 2.0]},
            {"name": "valid", "error": 1.0, "runtime": 2.0},
        ]
    )
    directions = {"error": "min", "runtime": "min"}

    indices = pareto_front_indices(table, ["error", "runtime"], directions=directions)
    mask = is_pareto_front(table, ["error", "runtime"], directions=directions)

    assert table.loc[indices, "name"].tolist() == ["valid"]
    assert mask.tolist() == [False, True]
