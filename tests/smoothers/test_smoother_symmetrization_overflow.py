from __future__ import annotations

import numpy as np
import numpy.testing as npt
from pyrecest.smoothers import RauchTungStriebelSmoother


def test_shared_smoother_symmetrization_avoids_intermediate_overflow() -> None:
    matrix = np.full((2, 2), np.finfo(np.float32).max, dtype=np.float32)

    with np.errstate(over="raise", invalid="raise"):
        result = RauchTungStriebelSmoother._symmetrize(matrix)

    result = np.asarray(result)
    npt.assert_array_equal(result, matrix)
    assert np.all(np.isfinite(result))
