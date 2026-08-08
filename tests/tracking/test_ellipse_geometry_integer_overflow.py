from __future__ import annotations

import numpy as np
import numpy.testing as npt
from pyrecest.tracking import project_symmetric_covariance


def test_covariance_projection_avoids_integer_overflow() -> None:
    covariance = np.array(
        [
            [120, 100],
            [80, 110],
        ],
        dtype=np.int8,
    )

    projected = np.asarray(project_symmetric_covariance(covariance))

    npt.assert_allclose(
        projected,
        np.array(
            [
                [120.0, 90.0],
                [90.0, 110.0],
            ]
        ),
    )
    assert np.issubdtype(projected.dtype, np.floating)
