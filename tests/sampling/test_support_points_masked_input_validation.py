from __future__ import annotations

import warnings

import numpy as np
import pytest
from pyrecest.sampling import (
    ellipsoid_axis_offsets,
    ellipsoid_sigma_points,
    mahalanobis_support_points,
    support_points_from_axis_offsets,
)


def test_support_point_arrays_reject_masked_values() -> None:
    masked_center = np.ma.array([0.0, 1.0], mask=[False, True])
    masked_covariance = np.ma.array(
        np.eye(2),
        mask=[[False, False], [False, True]],
    )

    with pytest.raises(ValueError, match="centers must contain real numeric values"):
        support_points_from_axis_offsets(masked_center, np.eye(2))
    with pytest.raises(
        ValueError,
        match="covariance must contain real numeric values",
    ):
        ellipsoid_axis_offsets(masked_covariance)


def test_nested_masked_direction_is_rejected_without_conversion_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(
            ValueError,
            match="directions must contain real numeric values",
        ):
            mahalanobis_support_points(
                [0.0, 0.0],
                np.eye(2),
                [[1.0, np.ma.masked]],
            )


def test_support_point_scalar_controls_reject_masked_values() -> None:
    masked_radius = np.ma.array(1.0, mask=True)
    masked_flag = np.ma.array(True, mask=True)

    with pytest.raises(ValueError, match="radius must be a finite non-negative scalar"):
        ellipsoid_axis_offsets(np.eye(2), radius=masked_radius)
    with pytest.raises(ValueError, match="include_center must be a boolean"):
        support_points_from_axis_offsets(
            [0.0, 0.0],
            np.eye(2),
            include_center=masked_flag,
        )
    with pytest.raises(ValueError, match="radius must be a finite non-negative scalar"):
        ellipsoid_sigma_points(
            [0.0, 0.0],
            np.eye(2),
            radii=(masked_radius,),
        )


def test_clear_mask_support_point_inputs_remain_supported() -> None:
    mean = np.ma.array([0.0, 0.0], mask=False)
    covariance = np.ma.array(np.eye(2), mask=False)
    directions = np.ma.array([[1.0, 0.0]], mask=False)
    radius = np.ma.array(1.0, mask=False)
    normalize = np.ma.array(True, mask=False)

    support = mahalanobis_support_points(
        mean,
        covariance,
        directions,
        radius=radius,
        normalize_directions=normalize,
    )

    np.testing.assert_allclose(support, [[1.0, 0.0]])
