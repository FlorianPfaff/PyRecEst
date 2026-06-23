from __future__ import annotations

import math

import pandas as pd
import pytest
from pyrecest.evaluation import constraint_mask


def test_constraint_mask_rejects_nonfinite_thresholds() -> None:
    table = pd.DataFrame([{"name": "candidate", "score": 1.0}])

    for threshold in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite scalar"):
            constraint_mask(table, {"score": ("!=", threshold)})


def test_constraint_mask_rejects_nonnumeric_thresholds() -> None:
    table = pd.DataFrame([{"name": "candidate", "score": 1.0}])

    with pytest.raises(ValueError, match="finite scalar"):
        constraint_mask(table, {"score": ("<=", "not-a-number")})
