import json

import numpy as np
from pyrecest.evaluation.diagnostic_summaries import top_residuals


def test_top_residuals_serializes_zero_dimensional_numpy_arrays():
    rows = top_residuals(
        [
            {
                "residual_norm": np.array(2.0),
                "backend_scalar": np.array(3.5),
                "backend_flag": np.array(True),
            }
        ],
        top_n=1,
    )

    assert rows == [
        {
            "residual_norm": 2.0,
            "backend_scalar": 3.5,
            "backend_flag": True,
        }
    ]
    json.dumps(rows)
