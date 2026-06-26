from __future__ import annotations

import pandas as pd
from pyrecest.evaluation import is_pareto_front, pareto_front_indices, record_dominates


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


def test_pareto_front_treats_numeric_text_objectives_as_missing() -> None:
    table = pd.DataFrame(
        [
            {"name": "text_fast", "runtime": "0.0"},
            {"name": "numeric_slow", "runtime": 1.0},
        ]
    )

    indices = pareto_front_indices(table, ["runtime"], directions={"runtime": "min"})
    mask = is_pareto_front(table, ["runtime"], directions={"runtime": "min"})

    assert table.loc[indices, "name"].tolist() == ["text_fast", "numeric_slow"]
    assert mask.to_numpy().all()
