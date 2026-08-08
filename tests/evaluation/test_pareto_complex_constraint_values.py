from __future__ import annotations

import numpy as np
import pandas as pd
from pyrecest.evaluation import constraint_mask


def test_constraint_mask_treats_complex_dtype_columns_as_missing() -> None:
    table = pd.DataFrame(
        {
            "score": np.asarray(
                [0.25 + 4.0j, 0.25 + 0.0j],
                dtype=np.complex128,
            )
        }
    )

    mask = constraint_mask(table, {"score": ("<=", 0.5)})

    assert mask.tolist() == [False, False]


def test_constraint_mask_treats_object_complex_scalars_as_missing() -> None:
    table = pd.DataFrame(
        {
            "score": pd.Series(
                [
                    0.25,
                    np.complex64(0.25 + 4.0j),
                    np.asarray(0.25 + 0.0j),
                ],
                dtype=object,
            )
        }
    )

    mask = constraint_mask(table, {"score": ("<=", 0.5)})

    assert mask.tolist() == [True, False, False]
