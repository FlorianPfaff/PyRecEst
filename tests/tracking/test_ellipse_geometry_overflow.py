from __future__ import annotations

import numpy as np
import numpy.testing as npt

from pyrecest.tracking import symmetrize


def test_symmetrize_large_finite_entries_without_intermediate_overflow() -> None:
    matrix = np.full((2, 2), 1.0e308)

    with np.errstate(over="raise", invalid="raise"):
        result = symmetrize(matrix)

    npt.assert_array_equal(result, matrix)
    assert np.all(np.isfinite(result))
