import pandas as pd
import pytest
from pyrecest.evaluation.pareto import is_pareto_front, pareto_front_indices


def test_pareto_front_indices_reject_missing_objective_column():
    table = pd.DataFrame({"cost": [1.0, 2.0]})

    with pytest.raises(
        ValueError, match="Pareto objective columns are missing: quality"
    ):
        pareto_front_indices(
            table,
            ["cost", "quality"],
            directions={"cost": "min", "quality": "max"},
        )


def test_is_pareto_front_rejects_missing_objective_column():
    table = pd.DataFrame({"cost": [1.0, 2.0]})

    with pytest.raises(
        ValueError, match="Pareto objective columns are missing: quality"
    ):
        is_pareto_front(
            table,
            ["cost", "quality"],
            directions={"cost": "min", "quality": "max"},
        )


def test_pareto_front_indices_allow_empty_table_when_objective_columns_exist():
    table = pd.DataFrame({"cost": pd.Series(dtype=float)})

    assert pareto_front_indices(table, ["cost"], directions={"cost": "min"}) == []
