from __future__ import annotations

import numpy as np
import numpy.testing as npt

from pyrecest.tracking import ellipse_extent_matrix, extent_matrix_from_shape


def test_extent_matrix_is_invariant_to_axis_sign_representation() -> None:
    theta = 0.3
    positive_axes = np.array([3.0, 1.0])
    signed_axes = np.array([-3.0, -1.0])

    expected = ellipse_extent_matrix(theta, positive_axes)

    npt.assert_allclose(
        ellipse_extent_matrix(theta, signed_axes),
        expected,
        atol=1e-12,
    )
    npt.assert_allclose(
        extent_matrix_from_shape(np.array([theta, *signed_axes])),
        expected,
        atol=1e-12,
    )
