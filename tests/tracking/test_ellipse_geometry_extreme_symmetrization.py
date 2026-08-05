from __future__ import annotations

import numpy as np
import numpy.testing as npt
from pyrecest.tracking.ellipse_geometry import (
    project_symmetric_covariance,
    symmetrize,
)


def test_symmetrize_preserves_extreme_finite_diagonal() -> None:
    covariance = np.diag([1.0e308, 2.0e307])

    with np.errstate(over="raise", invalid="raise"):
        symmetric = np.asarray(symmetrize(covariance))

    assert np.all(np.isfinite(symmetric))
    npt.assert_array_equal(symmetric, covariance)


def test_covariance_projection_preserves_extreme_finite_diagonal() -> None:
    covariance = np.diag([1.0e308, 2.0e307])

    with np.errstate(over="raise", invalid="raise"):
        projected = np.asarray(project_symmetric_covariance(covariance))

    assert np.all(np.isfinite(projected))
    npt.assert_array_equal(projected, covariance)
