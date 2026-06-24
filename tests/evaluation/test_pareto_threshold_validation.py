from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pyrecest.evaluation import constraint_mask


def test_constraint_threshold_rejects_bool_and_non_scalar_values() -> None:
    table = pd.DataFrame([{"score": 1.0}])
    invalid_thresholds = (True, np.array([1.0]))

    for threshold in invalid_thresholds:
        with pytest.raises(
            ValueError,
            match="Constraint threshold for 'score' must be a finite scalar",
        ):
            constraint_mask(table, {"score": ("<=", threshold)})


def test_constraint_threshold_accepts_scalar_numpy_float() -> None:
    table = pd.DataFrame(
        [{"name": "keep", "score": 1.0}, {"name": "drop", "score": 2.0}]
    )

    mask = constraint_mask(table, {"score": ("<=", np.array(1.0))})

    assert table.loc[mask, "name"].tolist() == ["keep"]
