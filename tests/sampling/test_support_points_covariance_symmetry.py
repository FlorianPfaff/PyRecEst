from __future__ import annotations

import numpy as np
import pytest
from pyrecest.sampling import (
    ellipsoid_axis_offsets,
    ellipsoid_sigma_points,
    mahalanobis_support_points,
)


@pytest.mark.parametrize(
    "bad_covariance",
    [
        np.asarray([[1.0, 0.25], [0.5, 1.0]]),
        np.asarray(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[2.0, -0.1], [0.1, 2.0]],
            ]
        ),
    ],
)
def test_ellipsoid_axis_offsets_reject_nonsymmetric_covariance(
    bad_covariance: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="symmetric"):
        ellipsoid_axis_offsets(bad_covariance)


@pytest.mark.parametrize(
    "factory",
    [
        lambda covariance: ellipsoid_sigma_points([0.0, 0.0], covariance),
        lambda covariance: mahalanobis_support_points(
            [0.0, 0.0], covariance, [[1.0, 0.0]]
        ),
    ],
)
def test_covariance_ellipsoid_helpers_reject_nonsymmetric_covariance(factory) -> None:
    bad_covariance = np.asarray([[1.0, 0.25], [0.5, 1.0]])

    with pytest.raises(ValueError, match="symmetric"):
        factory(bad_covariance)
