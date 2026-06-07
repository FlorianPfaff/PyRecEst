from __future__ import annotations

import numpy as np
import numpy.testing as npt
from pyrecest.tracking import (
    canonicalize_ellipse_axes,
    canonicalize_ellipse_shape,
    ellipse_angle_delta,
    ellipse_extent_matrix,
    ellipse_shape_canonicalization_transform,
    shape_from_extent_matrix,
    wrap_ellipse_angle_to_reference,
)


def test_ellipse_angle_delta_is_pi_periodic() -> None:
    assert np.isclose(float(ellipse_angle_delta(0.0, np.pi - 0.1)), -0.1)
    assert np.isclose(float(wrap_ellipse_angle_to_reference(0.0, np.pi - 0.1)), -0.1)


def test_canonicalize_ellipse_shape_transforms_full_shape_covariance() -> None:
    shape = np.array([0.2, -1.0, 2.0])
    covariance = np.array(
        [
            [4.0, 0.5, -0.25],
            [0.5, 9.0, 1.5],
            [-0.25, 1.5, 16.0],
        ],
        dtype=float,
    )

    canonical_shape, canonical_covariance = canonicalize_ellipse_shape(
        shape,
        covariance,
        major_axis_first=True,
        reference_orientation=shape[0],
    )

    sign_transform = np.diag([1.0, -1.0, 1.0])
    swap_transform = np.eye(3)
    swap_transform[[1, 2], :] = swap_transform[[2, 1], :]
    expected_covariance = swap_transform @ sign_transform @ covariance
    expected_covariance = expected_covariance @ sign_transform.T @ swap_transform.T

    npt.assert_allclose(canonical_shape[1:], np.array([2.0, 1.0]))
    assert np.isclose(
        float(ellipse_angle_delta(shape[0] + np.pi / 2.0, canonical_shape[0])),
        0.0,
    )
    npt.assert_allclose(canonical_covariance, expected_covariance)


def test_canonicalization_transform_can_be_embedded_in_larger_covariance() -> None:
    mean = np.array([5.0, 0.2, -1.0, 2.0])
    covariance = np.eye(4)
    covariance[0, 2] = covariance[2, 0] = 0.3
    covariance[2, 3] = covariance[3, 2] = -0.4

    canonical_shape, shape_transform = ellipse_shape_canonicalization_transform(
        mean[1:],
        major_axis_first=True,
    )
    transform = np.eye(4)
    transform[1:, 1:] = np.asarray(shape_transform, dtype=float)
    canonical_covariance = transform @ covariance @ transform.T

    sign_transform = np.eye(4)
    sign_transform[2, 2] = -1.0
    swap_transform = np.eye(4)
    swap_transform[[2, 3], :] = swap_transform[[3, 2], :]
    expected_covariance = swap_transform @ sign_transform @ covariance
    expected_covariance = expected_covariance @ sign_transform.T @ swap_transform.T

    npt.assert_allclose(canonical_shape[1:], np.array([2.0, 1.0]))
    npt.assert_allclose(canonical_covariance, expected_covariance)


def test_canonicalize_ellipse_axes_matches_shape_axis_block() -> None:
    axes, axis_covariance, swapped = canonicalize_ellipse_axes(
        np.array([-1.0, 2.0]),
        np.array([[9.0, 1.5], [1.5, 16.0]]),
        major_axis_first=True,
    )

    npt.assert_allclose(axes, np.array([2.0, 1.0]))
    npt.assert_allclose(axis_covariance, np.array([[16.0, -1.5], [-1.5, 9.0]]))
    assert swapped


def test_extent_matrix_is_invariant_to_axis_swap_representation() -> None:
    theta = 0.3
    axes = np.array([3.0, 1.0])

    extent = ellipse_extent_matrix(theta, axes)
    swapped_extent = ellipse_extent_matrix(theta + np.pi / 2.0, axes[::-1].copy())

    npt.assert_allclose(extent, swapped_extent, atol=1e-12)


def test_shape_from_extent_matrix_orders_axes_and_respects_reference() -> None:
    extent = ellipse_extent_matrix(np.pi - 0.2, np.array([4.0, 1.5]))

    shape = shape_from_extent_matrix(extent, reference_orientation=-0.2)
