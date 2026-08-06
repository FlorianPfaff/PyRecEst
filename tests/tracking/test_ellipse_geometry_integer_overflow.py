from __future__ import annotations

import numpy as np
import numpy.testing as npt
from pyrecest.tracking.ellipse_geometry import symmetrize


def test_symmetrize_avoids_integer_overflow() -> None:
    matrix = np.array(
        [
            [120, 100],
            [80, 110],
        ],
        dtype=np.int8,
    )

    result = np.asarray(symmetrize(matrix))

    npt.assert_allclose(
        result,
        np.array(
            [
                [120.0, 90.0],
                [90.0, 110.0],
            ]
        ),
    )
    assert np.issubdtype(result.dtype, np.floating)
