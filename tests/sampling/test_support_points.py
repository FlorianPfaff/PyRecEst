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


def test_support_points_from_axis_offsets_can_omit_center() -> None:
    support = support_points_from_axis_offsets(
        [0.0, 0.0], np.eye(2), include_center=False
    )

    assert support.shape == (4, 2)
    assert np.allclose(support, [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])


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


@pytest.mark.parametrize(
    "bad_radius", ["1.0", b"1.0", np.str_("1.0"), np.bytes_(b"1.0")]
)
def test_ellipsoid_axis_offsets_reject_text_radius(bad_radius: object) -> None:
    with pytest.raises(ValueError, match="radius"):
        ellipsoid_axis_offsets(np.eye(2), radius=bad_radius)


@pytest.mark.parametrize(
    "bad_flag",
    ["False", 1, np.asarray([True]), None],
)
def test_support_points_reject_non_boolean_include_center(bad_flag: object) -> None:
    with pytest.raises(ValueError, match="include_center"):
        support_points_from_axis_offsets([0.0, 0.0], np.eye(2), include_center=bad_flag)


@pytest.mark.parametrize(
    "keyword",
    ["sort_descending", "clip_negative_eigenvalues"],
)
def test_ellipsoid_axis_offsets_reject_non_boolean_flags(keyword: str) -> None:
    with pytest.raises(ValueError, match=keyword):
        ellipsoid_axis_offsets(np.eye(2), **{keyword: "False"})


def test_ellipsoid_sigma_points_reject_non_boolean_include_center() -> None:
    with pytest.raises(ValueError, match="include_center"):
        ellipsoid_sigma_points([0.0, 0.0], np.eye(2), include_center="False")


def test_mahalanobis_support_points_reject_non_boolean_normalize_flag() -> None:
    with pytest.raises(ValueError, match="normalize_directions"):
        mahalanobis_support_points(
            [0.0, 0.0], np.eye(2), [[1.0, 0.0]], normalize_directions="False"
        )


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


@pytest.mark.parametrize(
    "bad_centers",
    [
        [True, False],
        np.asarray([True, False]),
        ["0.0", "1.0"],
    ],
)
def test_support_points_reject_non_real_numeric_centers(bad_centers: object) -> None:
    with pytest.raises(ValueError, match="centers"):
        support_points_from_axis_offsets(bad_centers, np.eye(2))


@pytest.mark.parametrize(
    "bad_centers",
    [
        np.asarray([np.datetime64("2026-01-01")]),
        np.asarray([np.datetime64("2026-01-01")], dtype=object),
    ],
)
def test_support_points_reject_temporal_centers(bad_centers: object) -> None:
    with pytest.raises(ValueError, match="centers"):
        support_points_from_axis_offsets(bad_centers, np.eye(1))


@pytest.mark.parametrize(
    "bad_covariance",
    [
        [["1.0", "0.0"], ["0.0", "1.0"]],
        [[True, False], [False, True]],
    ],
)
def test_ellipsoid_axis_offsets_reject_non_real_numeric_covariance(
    bad_covariance: object,
) -> None:
    with pytest.raises(ValueError, match="covariance"):
        ellipsoid_axis_offsets(bad_covariance)


@pytest.mark.parametrize(
    "bad_covariance",
    [
        np.asarray([[np.timedelta64(3, "s")]]),
        np.asarray([[np.timedelta64(3, "s")]], dtype=object),
    ],
)
def test_ellipsoid_axis_offsets_reject_temporal_covariance(
    bad_covariance: object,
) -> None:
    with pytest.raises(ValueError, match="covariance"):
        ellipsoid_axis_offsets(bad_covariance)


def test_support_points_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="axis_offsets"):
        support_points_from_axis_offsets([0.0, 0.0, 0.0], np.eye(2))


def test_mahalanobis_support_points_reject_zero_direction() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        mahalanobis_support_points([0.0, 0.0], np.eye(2), [[0.0, 0.0]])
