from __future__ import annotations

import numpy as np
import pytest
from pyrecest.sampling import (
    ellipsoid_axis_offsets,
    ellipsoid_axis_support_points,
    ellipsoid_sigma_points,
    mahalanobis_support_points,
    projected_linear_variance_from_axis_offsets,
    support_points_from_axis_offsets,
)


def test_ellipsoid_axis_support_points_for_diagonal_covariance() -> None:
    support = ellipsoid_axis_support_points([1.0, 2.0], np.diag([4.0, 1.0]))

    assert support.shape == (5, 2)
    assert np.allclose(
        support,
        np.asarray(
            [
                [1.0, 2.0],
                [3.0, 2.0],
                [-1.0, 2.0],
                [1.0, 3.0],
                [1.0, 1.0],
            ]
        ),
    )


def test_support_points_from_axis_offsets_supports_batches() -> None:
    centers = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    axis_offsets = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 3.0]],
        ],
        dtype=np.float64,
    )

    support = support_points_from_axis_offsets(centers, axis_offsets)

    assert support.shape == (2, 5, 2)
    assert np.allclose(
        support[0], [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    )
    assert np.allclose(
        support[1], [[1.0, 1.0], [3.0, 1.0], [-1.0, 1.0], [1.0, 4.0], [1.0, -2.0]]
    )


def test_projected_linear_variance_from_axis_offsets() -> None:
    axis_offsets = np.broadcast_to(np.eye(2, dtype=np.float64), (2, 2, 2))
    coefficients = np.asarray([[3.0, 4.0], [1.0, 2.0]], dtype=np.float64)

    variance = projected_linear_variance_from_axis_offsets(coefficients, axis_offsets)

    assert np.allclose(variance, [25.0, 5.0])
    assert projected_linear_variance_from_axis_offsets(
        [3.0, 4.0], np.eye(2)
    ) == pytest.approx(25.0)


def test_ellipsoid_sigma_points_include_center_once_for_multiple_radii() -> None:
    support = ellipsoid_sigma_points([0.0, 0.0], np.eye(2), radii=(1.0, 2.0))

    assert support.shape == (9, 2)
    assert np.allclose(support[0], [0.0, 0.0])
    assert np.isclose(np.max(np.linalg.norm(support[1:], axis=1)), 2.0)


def test_mahalanobis_support_points_use_world_directions() -> None:
    support = mahalanobis_support_points(
        [0.0, 0.0], np.diag([4.0, 1.0]), [[1.0, 0.0], [0.0, 1.0]]
    )

    assert support.shape == (2, 2)
    assert np.allclose(support, [[2.0, 0.0], [0.0, 1.0]])


def test_ellipsoid_axis_offsets_reject_negative_radius() -> None:
    with pytest.raises(ValueError, match="radius"):
        ellipsoid_axis_offsets(np.eye(2), radius=-1.0)


@pytest.mark.parametrize("bad_radius", [np.nan, np.inf, -np.inf])
def test_ellipsoid_axis_offsets_reject_nonfinite_radius(bad_radius: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        ellipsoid_axis_offsets(np.eye(2), radius=bad_radius)


@pytest.mark.parametrize("bad_radius", [np.nan, np.inf, -np.inf])
def test_ellipsoid_sigma_points_reject_nonfinite_radii(bad_radius: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        ellipsoid_sigma_points([0.0, 0.0], np.eye(2), radii=(1.0, bad_radius))


@pytest.mark.parametrize("bad_radius", [np.nan, np.inf, -np.inf])
def test_mahalanobis_support_points_reject_nonfinite_radius(bad_radius: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        mahalanobis_support_points(
            [0.0, 0.0], np.eye(2), [[1.0, 0.0]], radius=bad_radius
        )


def test_support_points_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="axis_offsets"):
        support_points_from_axis_offsets([0.0, 0.0, 0.0], np.eye(2))


def test_mahalanobis_support_points_reject_zero_direction() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        mahalanobis_support_points([0.0, 0.0], np.eye(2), [[0.0, 0.0]])
