from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _as_real_array(name: str, value: np.ndarray | Sequence[float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _as_point_batch(
    name: str, points: np.ndarray | Sequence[float]
) -> tuple[np.ndarray, bool]:
    array = _as_real_array(name, points)
    if array.ndim == 1:
        if array.shape[0] < 1:
            raise ValueError(f"{name} must have at least one dimension.")
        return array.reshape(1, -1), True
    if array.ndim == 2:
        if array.shape[0] < 1 or array.shape[1] < 1:
            raise ValueError(
                f"{name} must be non-empty and have at least one dimension."
            )
        return array, False
    raise ValueError(f"{name} must have shape (dim,) or (count, dim).")


def _as_covariance_batch(
    name: str, covariance: np.ndarray | Sequence[Sequence[float]]
) -> tuple[np.ndarray, bool]:
    array = _as_real_array(name, covariance)
    if array.ndim == 2:
        if array.shape[0] != array.shape[1] or array.shape[0] < 1:
            raise ValueError(f"{name} must be a non-empty square matrix.")
        return array.reshape(1, array.shape[0], array.shape[1]), True
    if array.ndim == 3:
        if array.shape[0] < 1 or array.shape[1] != array.shape[2] or array.shape[1] < 1:
            raise ValueError(
                f"{name} must have shape (count, dim, dim) with square matrices."
            )
        return array, False
    raise ValueError(f"{name} must have shape (dim, dim) or (count, dim, dim).")


def _as_axis_offset_batch(
    axis_offsets: np.ndarray | Sequence[Sequence[float]], *, dim: int
) -> tuple[np.ndarray, bool]:
    array = _as_real_array("axis_offsets", axis_offsets)
    if array.ndim == 2:
        if array.shape != (dim, dim):
            raise ValueError(
                f"axis_offsets must have shape ({dim}, {dim}) for a single center."
            )
        return array.reshape(1, dim, dim), True
    if array.ndim == 3:
        if array.shape[1:] != (dim, dim):
            raise ValueError(f"axis_offsets must have shape (count, {dim}, {dim}).")
        return array, False
    raise ValueError("axis_offsets must have shape (dim, dim) or (count, dim, dim).")


def _broadcast_batches(
    points: np.ndarray, point_single: bool, axis_offsets: np.ndarray, axis_single: bool
) -> tuple[np.ndarray, np.ndarray, bool]:
    if point_single and axis_single:
        return points, axis_offsets, True
    if point_single:
        return (
            np.broadcast_to(points, (axis_offsets.shape[0], points.shape[1])).copy(),
            axis_offsets,
            False,
        )
    if axis_single:
        return (
            points,
            np.broadcast_to(
                axis_offsets,
                (points.shape[0], axis_offsets.shape[1], axis_offsets.shape[2]),
            ).copy(),
            False,
        )
    if points.shape[0] != axis_offsets.shape[0]:
        raise ValueError(
            "point and axis-offset batches must have the same length or one batch must be singular."
        )
    return points, axis_offsets, False


def ellipsoid_axis_offsets(
    covariance: np.ndarray | Sequence[Sequence[float]],
    *,
    radius: float = 1.0,
    sort_descending: bool = True,
    clip_negative_eigenvalues: bool = True,
) -> np.ndarray:
    """Return principal-axis offset vectors for covariance ellipsoids.

    The returned matrix has the same convention as a covariance square root: each
    column is one principal-axis offset vector.  For a single covariance, the
    shape is ``(dim, dim)``.  For a batch, the shape is ``(count, dim, dim)``.
    """

    if radius < 0.0:
        raise ValueError("radius must be non-negative.")
    covariances, single = _as_covariance_batch("covariance", covariance)
    eigenvalues, eigenvectors = np.linalg.eigh(covariances)
    if sort_descending:
        order = np.argsort(eigenvalues, axis=1)[:, ::-1]
        eigenvalues = np.take_along_axis(eigenvalues, order, axis=1)
        eigenvectors = np.take_along_axis(eigenvectors, order[:, None, :], axis=2)
    if clip_negative_eigenvalues:
        eigenvalues = np.clip(eigenvalues, 0.0, None)
    elif bool(np.any(eigenvalues < 0.0)):
        raise ValueError("covariance has negative eigenvalues.")
    scales = np.sqrt(eigenvalues) * float(radius)
    offsets = eigenvectors * scales[:, None, :]
    return offsets[0] if single else offsets


def support_points_from_axis_offsets(
    centers: np.ndarray | Sequence[float],
    axis_offsets: np.ndarray | Sequence[Sequence[float]],
    *,
    include_center: bool = True,
) -> np.ndarray:
    """Return center and ± principal-axis support points.

    ``axis_offsets`` uses the convention that each column is one offset vector.
    Batched inputs return ``(count, support_point_count, dim)``; single inputs
    return ``(support_point_count, dim)``.
    """

    point_batch, point_single = _as_point_batch("centers", centers)
    axis_batch, axis_single = _as_axis_offset_batch(
        axis_offsets, dim=point_batch.shape[1]
    )
    point_batch, axis_batch, single = _broadcast_batches(
        point_batch, point_single, axis_batch, axis_single
    )

    pieces: list[np.ndarray] = []
    if include_center:
        pieces.append(point_batch)
    for axis in range(point_batch.shape[1]):
        offset = axis_batch[:, :, axis]
        pieces.append(point_batch + offset)
        pieces.append(point_batch - offset)
    if not pieces:
        pieces.append(point_batch)
    support = np.stack(pieces, axis=1)
    return support[0] if single else support


def ellipsoid_axis_support_points(
    mean: np.ndarray | Sequence[float],
    covariance: np.ndarray | Sequence[Sequence[float]],
    *,
    radius: float = 1.0,
    include_center: bool = True,
    sort_descending: bool = True,
    clip_negative_eigenvalues: bool = True,
) -> np.ndarray:
    """Return center and principal-axis support points of a covariance ellipsoid."""

    offsets = ellipsoid_axis_offsets(
        covariance,
        radius=radius,
        sort_descending=sort_descending,
        clip_negative_eigenvalues=clip_negative_eigenvalues,
    )
    return support_points_from_axis_offsets(
        mean, offsets, include_center=include_center
    )


def ellipsoid_sigma_points(
    mean: np.ndarray | Sequence[float],
    covariance: np.ndarray | Sequence[Sequence[float]],
    *,
    radii: Sequence[float] = (1.0,),
    include_center: bool = True,
    sort_descending: bool = True,
    clip_negative_eigenvalues: bool = True,
) -> np.ndarray:
    """Return support points at one or more Mahalanobis radii.

    The center is included at most once.  Single inputs return
    ``(support_point_count, dim)``; batched inputs return
    ``(count, support_point_count, dim)``.
    """

    radii_tuple = tuple(float(radius) for radius in radii)
    if not radii_tuple:
        raise ValueError("radii must contain at least one radius.")
    if any(radius < 0.0 for radius in radii_tuple):
        raise ValueError("radii must be non-negative.")

    centers, center_single = _as_point_batch("mean", mean)
    covariances, cov_single = _as_covariance_batch("covariance", covariance)
    if covariances.shape[1] != centers.shape[1]:
        raise ValueError("mean and covariance dimensions must agree.")
    if center_single and not cov_single:
        centers = np.broadcast_to(
            centers, (covariances.shape[0], centers.shape[1])
        ).copy()
        single = False
    elif cov_single and not center_single:
        covariances = np.broadcast_to(
            covariances, (centers.shape[0], covariances.shape[1], covariances.shape[2])
        ).copy()
        single = False
    elif center_single and cov_single:
        single = True
    else:
        if centers.shape[0] != covariances.shape[0]:
            raise ValueError(
                "mean and covariance batches must have the same length or one batch must be singular."
            )
        single = False

    pieces: list[np.ndarray] = []
    if include_center:
        pieces.append(centers)
    for radius in radii_tuple:
        offsets = ellipsoid_axis_offsets(
            covariances,
            radius=radius,
            sort_descending=sort_descending,
            clip_negative_eigenvalues=clip_negative_eigenvalues,
        )
        offsets = offsets.reshape(
            covariances.shape[0], centers.shape[1], centers.shape[1]
        )
        for axis in range(centers.shape[1]):
            offset = offsets[:, :, axis]
            pieces.append(centers + offset)
            pieces.append(centers - offset)
    support = np.stack(pieces, axis=1)
    return support[0] if single else support


def mahalanobis_support_points(
    mean: np.ndarray | Sequence[float],
    covariance: np.ndarray | Sequence[Sequence[float]],
    directions: np.ndarray | Sequence[Sequence[float]],
    *,
    radius: float = 1.0,
    normalize_directions: bool = True,
) -> np.ndarray:
    """Map unit directions to a covariance ellipsoid's surface."""

    if radius < 0.0:
        raise ValueError("radius must be non-negative.")
    centers, center_single = _as_point_batch("mean", mean)
    covariances, cov_single = _as_covariance_batch("covariance", covariance)
    directions_array = _as_real_array("directions", directions)
    if directions_array.ndim == 1:
        directions_array = directions_array.reshape(1, -1)
    if directions_array.ndim != 2 or directions_array.shape[1] != centers.shape[1]:
        raise ValueError("directions must have shape (direction_count, dim).")
    if normalize_directions:
        norms = np.linalg.norm(directions_array, axis=1, keepdims=True)
        if bool(np.any(norms <= 0.0)):
            raise ValueError(
                "directions must be non-zero when normalize_directions=True."
            )
        directions_array = directions_array / norms

    if covariances.shape[1] != centers.shape[1]:
        raise ValueError("mean and covariance dimensions must agree.")
    if center_single and not cov_single:
        centers = np.broadcast_to(
            centers, (covariances.shape[0], centers.shape[1])
        ).copy()
        single = False
    elif cov_single and not center_single:
        covariances = np.broadcast_to(
            covariances, (centers.shape[0], covariances.shape[1], covariances.shape[2])
        ).copy()
        single = False
    elif center_single and cov_single:
        single = True
    else:
        if centers.shape[0] != covariances.shape[0]:
            raise ValueError(
                "mean and covariance batches must have the same length or one batch must be singular."
            )
        single = False

    eigenvalues, eigenvectors = np.linalg.eigh(covariances)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    sqrt_covariances = np.einsum(
        "bij,bj,bkj->bik",
        eigenvectors,
        np.sqrt(eigenvalues) * float(radius),
        eigenvectors,
    )
    mapped_offsets = np.einsum("mj,bji->bmi", directions_array, sqrt_covariances)
    points = centers[:, None, :] + mapped_offsets
    return points[0] if single else points


def projected_linear_variance_from_axis_offsets(
    linear_coefficients: np.ndarray | Sequence[float],
    axis_offsets: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray | float:
    """Return ``sum_i (a^T u_i)^2`` for ellipsoid axis offsets ``u_i``.

    This is the first-order variance of a linearized scalar function with gradient
    ``a`` over an ellipsoid whose square-root covariance columns are ``u_i``.
    """

    coefficients, coeff_single = _as_point_batch(
        "linear_coefficients", linear_coefficients
    )
    axes, axis_single = _as_axis_offset_batch(axis_offsets, dim=coefficients.shape[1])
    coefficients, axes, single = _broadcast_batches(
        coefficients, coeff_single, axes, axis_single
    )
    projections = np.einsum("bi,bij->bj", coefficients, axes)
    variances = np.sum(np.square(projections), axis=1)
    return float(variances[0]) if single else variances
