from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pyrecest.evaluation import constraint_mask


def test_constraint_mask_rejects_malformed_constraint_specs() -> None:
    table = pd.DataFrame([{"name": "a", "score": 1.0}])
    invalid_specs = (
        1,
        object(),
        "<=",
        np.array(["<=", 1.0, 2.0], dtype=object),
    )

    for spec in invalid_specs:
        with pytest.raises(ValueError, match="Constraint"):
            constraint_mask(table, {"score": spec})  # type: ignore[arg-type]
