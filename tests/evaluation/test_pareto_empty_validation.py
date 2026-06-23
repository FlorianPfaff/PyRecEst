from __future__ import annotations

import pandas as pd
import pytest
from pyrecest.evaluation import pareto_front_indices


def test_pareto_front_indices_validates_directions_on_empty_table() -> None:
    table = pd.DataFrame(columns=["error"])

    with pytest.raises(ValueError, match="Invalid objective directions"):
        pareto_front_indices(
            table,
            ["error"],
            directions={"error": "sideways"},  # type: ignore[dict-item]
        )
