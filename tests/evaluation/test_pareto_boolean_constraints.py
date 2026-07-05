from __future__ import annotations

import pandas as pd
from pyrecest.evaluation import constraint_mask


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
