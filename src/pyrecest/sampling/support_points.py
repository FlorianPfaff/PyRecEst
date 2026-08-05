from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta

import numpy as np

_TEXT_TYPES = (str, bytes, bytearray, np.str_, np.bytes_)
_BOOLEAN_TYPES = (bool, np.bool_)
_COMPLEX_TYPES = (complex, np.complexfloating)
_TEMPORAL_TYPES = (date, datetime, timedelta, np.datetime64, np.timedelta64)


def _contains_values_of_type(value: object, types: tuple[type, ...]) -> bool:
    if isinstance(value, types):
        return True
    try:
        values = np.asarray(value, dtype=object).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return False
    return any(isinstance(item, types) for item in values)


def _has_temporal_dtype(value: object) -> bool:
    try:
        return np.asarray(value).dtype.kind in "Mm"
    except (TypeError, ValueError, RuntimeError):
        return False


def _has_non_real_numeric_values(value: object) -> bool:
    return (
        _has_temporal_dtype(value)
        or _contains_values_of_type(value, _TEXT_TYPES)
        or _contains_values_of_type(value, _BOOLEAN_TYPES)
        or _contains_values_of_type(value, _COMPLEX_TYPES)
        or _contains_values_of_type(value, _TEMPORAL_TYPES)
    )


def _as_real_array(name: str, value: np.ndarray | Sequence[float]) -> np.ndarray:
    if _has_non_real_numeric_values(value):
        raise ValueError(f"{name} must contain real numeric values.")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values.") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _as_finite_nonnegative_scalar(name: str, value: float) -> float:
    message = f"{name} must be a finite non-negative scalar."
    if _has_non_real_numeric_values(value):
        raise ValueError(message)
    array = np.asarray(value)
    if array.shape != () or array.dtype == np.bool_:
        raise ValueError(message)
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(message)
    try:
        scalar_float = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(scalar_float) or scalar_float < 0.0:
        raise ValueError(message)
    return scalar_float


def _as_bool_scalar(name: str, value: bool) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a boolean.") from exc
    if array.shape == () and array.dtype == np.bool_:
        return bool(array.item())
    raise ValueError(f"{name} must be a boolean.")


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


def _ensure_symmetric_covariance_batch(name: str, array: np.ndarray) -> np.ndarray:
    if not np.allclose(array, np.swapaxes(array, -1, -2), rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric.")
    return array


def _as_covariance_batch(
    name: str, covariance: np.ndarray | Sequence[Sequence[float]]
) -> tuple[np.ndarray, bool]:
    array = _as_real_array(name, covariance)
    if array.ndim == 2:
        if array.shape[0] != array.shape[1] or array.shape[0] < 1:
            raise ValueError(f"{name} must be a non-empty square matrix.")
        array = _ensure_symmetric_covariance_batch(name, array)
        return array.reshape(1, array.shape[0], array.shape[1]), True
    if array.ndim == 3:
        if array.shape[0] < 1 or array.shape[1] != array.shape[2] or array.shape[1] < 1:
            raise ValueError(
                f"{name} must have shape (count, dim, dim) with square matrices."
            )
        array = _ensure_symmetric_covariance_batch(name, array)
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

    radius = _as_finite_nonnegative_scalar("radius", radius)
    sort_descending = _as_bool_scalar("sort_descending", sort_descending)
    clip_negative_eigenvalues = _as_bool_scalar(
        "clip_negative_eigenvalues", clip_negative_eigenvalues
    )
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
    scales = np.sqrt(eigenvalues) * radius
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

    include_center = _as_bool_scalar("include_center", include_center)
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
    ``(support_point_count, dim)``; batched inputs return ``(count, support_point_count, dim)``.
    """

    include_center = _as_bool_scalar("include_center", include_center)
    sort_descending = _as_bool_scalar("sort_descending", sort_descending)
    clip_negative_eigenvalues = _as_bool_scalar(
        "clip_negative_eigenvalues", clip_negative_eigenvalues
    )
    radii_tuple = tuple(
        _as_finite_nonnegative_scalar("radius", radius) for radius in radii
    )
    if not radii_tuple:
        raise ValueError("radii must contain at least one radius.")

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
    zero_radius_requested = any(radius == 0.0 for radius in radii_tuple)
    if include_center or zero_radius_requested:
        pieces.append(centers)
    for radius in radii_tuple:
        if radius == 0.0:
            continue
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
    """Intersect world-frame direction rays with a covariance ellipsoid."""

    radius = _as_finite_nonnegative_scalar("radius", radius)
    normalize_directions = _as_bool_scalar("normalize_directions", normalize_directions)
    centers, center_single = _as_point_batch("mean", mean)
    covariances, cov_single = _as_covariance_batch("covariance", covariance)
    directions_array = _as_real_array("directions", directions)
    if directions_array.ndim == 1:
        directions_array = directions_array.reshape(1, -1)
    if directions_array.ndim != 2 or directions_array.shape[1] != centers.shape[1]:
        raise ValueError("directions must have shape (direction_count, dim).")
    if normalize_directions:
        direction_scales = np.max(
            np.abs(directions_array), axis=1, keepdims=True
        )
        if bool(np.any(direction_scales <= 0.0)):
            raise ValueError(
                "directions must be non-zero when normalize_directions=True."
            )
        scaled_directions = directions_array / direction_scales
        norms = np.linalg.norm(scaled_directions, axis=1, keepdims=True)
        directions_array = scaled_directions / norms

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
    positive_eigenvalues = eigenvalues > 0.0
    direction_components = np.einsum(
        "mi,bij->bmj", directions_array, eigenvectors
    )
    supported_components = np.where(
        positive_eigenvalues[:, None, :], direction_components, 0.0
    )
    supported_directions = np.einsum(
        "bmj,bij->bmi", supported_components, eigenvectors
    )

    inverse_eigenvalues = np.zeros_like(eigenvalues)
    np.divide(
        1.0,
        eigenvalues,
        out=inverse_eigenvalues,
        where=positive_eigenvalues,
    )
    squared_mahalanobis_norms = np.einsum(
        "bmj,bj,bmj->bm",
        supported_components,
        inverse_eigenvalues,
        supported_components,
    )
    ray_scales = np.zeros_like(squared_mahalanobis_norms)
    np.divide(
        radius,
        np.sqrt(squared_mahalanobis_norms),
        out=ray_scales,
        where=squared_mahalanobis_norms > 0.0,
    )
    mapped_offsets = supported_directions * ray_scales[:, :, None]
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
