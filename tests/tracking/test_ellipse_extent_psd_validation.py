"""Regression tests for ellipse extent positive-semidefinite validation."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from pyrecest.tracking import shape_from_extent_matrix


@pytest.mark.parametrize(
    "extent",
    (
        np.diag([4.0, -0.25]),
        np.diag([-1.0, -0.5]),
    ),
)
def test_shape_from_extent_matrix_rejects_indefinite_extent(extent) -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        shape_from_extent_matrix(extent)


def test_shape_from_extent_matrix_tolerates_roundoff_negative_eigenvalue() -> None:
    shape = shape_from_extent_matrix(
        np.diag([4.0, -1.0e-14]),
        minimum_axis_length=0.1,
    )

    npt.assert_allclose(np.asarray(shape)[1:], np.array([2.0, 0.1]))
