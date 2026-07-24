"""Geometry helpers for 2-D elliptical extended-object states.

The canonical MEM-style ellipse shape vector is
``[orientation, semi_axis_1, semi_axis_2]``.  Several EOT filters also expose a
public full-axis convention by doubling the two semi-axis entries.  This module
keeps the common angle wrapping, axis-sign, axis-swap, and covariance-transform
logic in one place so individual trackers do not need to reimplement the same
representation bookkeeping.
"""

from __future__ import annotations

from math import isfinite as _isfinite_scalar

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import abs as backend_abs
from pyrecest.backend import all as backend_all
from pyrecest.backend import (
    arctan2,
    array,
    asarray,
    cos,
    diag,
    eye,
    isfinite,
    linalg,
    maximum,
    pi,
    sin,
    sqrt,
)

_INVALID_SCALAR_TYPES = (bool, str, bytes, bytearray)


def symmetrize(matrix):
    """Return the symmetric part of ``matrix``."""

    matrix = asarray(matrix)
    return 0.5 * (matrix + matrix.T)


def project_symmetric_covariance(covariance, minimum_eigenvalue=0.0):
    """Project a symmetric covariance matrix to the PSD cone.

    Parameters
    ----------
    covariance : array-like
        Square covariance matrix to regularize.
    minimum_eigenvalue : float, default=0.0
        Eigenvalue floor used after symmetrization.
    """

    minimum_eigenvalue = _coerce_nonnegative_finite_scalar(
        minimum_eigenvalue, "minimum_eigenvalue"
    )
    covariance = symmetrize(_coerce_finite_square_matrix(covariance, "covariance"))
    eigenvalues, eigenvectors = linalg.eigh(covariance)
    eigenvalues = maximum(eigenvalues, minimum_eigenvalue)
    return symmetrize((eigenvectors * eigenvalues) @ eigenvectors.T)


def _coerce_bool_flag(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    try:
        value_array = asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a bool") from exc
    if value_array.shape != ():
        raise ValueError(f"{name} must be a scalar bool")
    scalar = value_array.item()
    if isinstance(scalar, bool):
        return bool(scalar)
    raise ValueError(f"{name} must be a bool")


def _coerce_nonnegative_finite_scalar(value, name: str) -> float:
    message = f"{name} must be a finite non-negative scalar"
    if isinstance(value, _INVALID_SCALAR_TYPES):
        raise ValueError(message)
    try:
        value_array = asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if value_array.shape != ():
        raise ValueError(message)
    scalar = value_array.item()
    if isinstance(scalar, _INVALID_SCALAR_TYPES):
        raise ValueError(message)
    try:
        scalar_float = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not _isfinite_scalar(scalar_float) or scalar_float < 0.0:
        raise ValueError(message)
    return scalar_float


def _coerce_finite_square_matrix(value, name: str):
    matrix = asarray(value)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a finite square matrix")
    if not bool(backend_all(isfinite(matrix))):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _coerce_finite_matrix_shape(value, name: str, shape: tuple[int, int]):
    matrix = asarray(value)
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not bool(backend_all(isfinite(matrix))):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def rotation_matrix_2d(angle):
    """Return the 2-D rotation matrix for ``angle``."""

    return array(
        [
            [cos(angle), -sin(angle)],
            [sin(angle), cos(angle)],
        ]
    )


def ellipse_angle_delta(reference, theta):
    """Return the axial orientation residual ``theta - reference``.

    Ellipse orientations are pi-periodic: ``theta`` and ``theta + pi`` describe
    the same physical extent.  The returned residual lies in the principal axial
    interval ``[-pi/2, pi/2]`` up to floating-point boundary effects.
    """

    return 0.5 * arctan2(sin(2.0 * (theta - reference)), cos(2.0 * (theta - reference)))


def wrap_ellipse_angle_to_reference(reference, theta):
    """Return the representation of ``theta`` closest to ``reference`` modulo pi."""

    return reference + ellipse_angle_delta(reference, theta)


def ellipse_extent_matrix(orientation, semi_axes):
    """Return the SPD extent matrix encoded by ``orientation`` and semi-axes."""

    semi_axes = backend_abs(asarray(semi_axes).reshape(2))
    rotation = rotation_matrix_2d(orientation)
    return symmetrize(rotation @ diag(semi_axes**2) @ rotation.T)


def extent_matrix_from_shape(shape_state):
    """Return the SPD extent matrix for ``[orientation, semi_axis_1, semi_axis_2]``."""

    shape_state = asarray(shape_state).reshape(3)
    return ellipse_extent_matrix(shape_state[0], shape_state[1:])


def shape_from_extent_matrix(
    extent,
    *,
    reference_orientation=None,
    minimum_axis_length=0.0,
):
    """Convert a 2x2 SPD extent matrix to MEM-style ellipse shape coordinates.

    The returned axes are ordered with the larger semi-axis first.  When
    ``reference_orientation`` is supplied, the orientation is unwrapped to the
    nearest pi-equivalent representation of that reference.
    """

    minimum_axis_length = _coerce_nonnegative_finite_scalar(
        minimum_axis_length, "minimum_axis_length"
    )
    extent = symmetrize(_coerce_finite_matrix_shape(extent, "extent", (2, 2)))
    eigenvalues, eigenvectors = linalg.eigh(extent)
    if float(eigenvalues[1]) >= float(eigenvalues[0]):
        major_index = 1
        minor_index = 0
    else:
        major_index = 0
        minor_index = 1

    axis_floor_sq = minimum_axis_length**2
    major_axis = sqrt(maximum(eigenvalues[major_index], axis_floor_sq))
    minor_axis = sqrt(maximum(eigenvalues[minor_index], axis_floor_sq))
    major_vector = eigenvectors[:, major_index]
    orientation = arctan2(major_vector[1], major_vector[0])
    if reference_orientation is not None:
        orientation = wrap_ellipse_angle_to_reference(
            reference_orientation, orientation
        )
    return array([orientation, major_axis, minor_axis])


def ellipse_shape_canonicalization_transform(
    shape_state,
    *,
    minimum_axis_length=1e-9,
    major_axis_first=False,
    reference_orientation=None,
):
    """Canonicalize a MEM-style ellipse shape and return its covariance transform.

    Parameters
    ----------
    shape_state : array-like, shape (3,)
        Shape vector ``[orientation, semi_axis_1, semi_axis_2]``.
    minimum_axis_length : float, default=1e-9
        Lower bound applied to both semi-axis lengths.
    major_axis_first : bool, default=False
        If true, swap the two axes when the second axis is longer than the
        first, and add ``pi/2`` to the orientation.
    reference_orientation : float, optional
        If supplied, unwrap the final orientation to the nearest pi-equivalent
        representation of the reference.

    Returns
    -------
    canonical_shape_state : array-like, shape (3,)
        Canonicalized shape vector.
    transform : array-like, shape (3, 3)
        Linear transform to apply to a covariance block in the same local shape
        coordinates: ``P_canonical = transform @ P @ transform.T``.  Orientation
        unwrapping does not change the covariance transform.
    """

    shape_state = asarray(shape_state).reshape(3)
    minimum_axis_length = _coerce_nonnegative_finite_scalar(
        minimum_axis_length, "minimum_axis_length"
    )
    major_axis_first = _coerce_bool_flag(major_axis_first, "major_axis_first")
    orientation = shape_state[0]
    axes = shape_state[1:]
    transform = eye(3)

    sign_1 = -1.0 if float(axes[0]) < 0.0 else 1.0
    sign_2 = -1.0 if float(axes[1]) < 0.0 else 1.0
    if sign_1 < 0.0 or sign_2 < 0.0:
        sign_transform = diag(array([1.0, sign_1, sign_2]))
        transform = sign_transform @ transform
    axes = maximum(backend_abs(axes), minimum_axis_length)

    if major_axis_first and float(axes[1]) > float(axes[0]):
        swap_transform = array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        transform = swap_transform @ transform
        axes = array([axes[1], axes[0]])
        orientation = orientation + 0.5 * pi

    if reference_orientation is not None:
        orientation = wrap_ellipse_angle_to_reference(
            reference_orientation, orientation
        )
    return array([orientation, axes[0], axes[1]]), transform


def canonicalize_ellipse_shape(
    shape_state,
    shape_covariance=None,
    *,
    minimum_axis_length=1e-9,
    minimum_covariance_eigenvalue=0.0,
    major_axis_first=False,
    reference_orientation=None,
):
    """Canonicalize a MEM-style ellipse shape and optional shape covariance.

    This applies the same sign and swap transforms to the covariance that are
    applied to the shape state.  Cross-covariance terms inside the 3x3 shape
    covariance are preserved and transformed consistently.
    """

    canonical_shape, transform = ellipse_shape_canonicalization_transform(
        shape_state,
        minimum_axis_length=minimum_axis_length,
        major_axis_first=major_axis_first,
        reference_orientation=reference_orientation,
    )
    if shape_covariance is None:
        return canonical_shape, None

    shape_covariance = _coerce_finite_matrix_shape(
        shape_covariance,
        "shape_covariance",
        (3, 3),
    )
    canonical_covariance = transform @ symmetrize(shape_covariance) @ transform.T
    canonical_covariance = project_symmetric_covariance(
        canonical_covariance,
        minimum_eigenvalue=minimum_covariance_eigenvalue,
    )
    return canonical_shape, canonical_covariance


def canonicalize_ellipse_axes(
    semi_axes,
    axis_covariance=None,
    *,
    minimum_axis_length=1e-9,
    minimum_covariance_eigenvalue=0.0,
    major_axis_first=False,
):
    """Canonicalize only the two semi-axis coordinates and optional covariance.

    Returns ``(canonical_axes, canonical_axis_covariance, swapped)``.  The
    orientation shift associated with a swap is intentionally not returned here;
    callers that track orientation should use :func:`canonicalize_ellipse_shape`
    or :func:`ellipse_shape_canonicalization_transform` instead.
    """

    semi_axes = asarray(semi_axes).reshape(2)
    shape, transform = ellipse_shape_canonicalization_transform(
        array([0.0, semi_axes[0], semi_axes[1]]),
        minimum_axis_length=minimum_axis_length,
        major_axis_first=major_axis_first,
    )
    axes = shape[1:]
    axis_transform = transform[1:, 1:]
    swapped = bool(
        float(axis_transform[0, 1]) != 0.0 or float(axis_transform[1, 0]) != 0.0
    )
    if axis_covariance is None:
        return axes, None, swapped
    axis_covariance = _coerce_finite_matrix_shape(
        axis_covariance,
        "axis_covariance",
        (2, 2),
    )
    canonical_axis_covariance = (
        axis_transform @ symmetrize(axis_covariance) @ axis_transform.T
    )
    canonical_axis_covariance = project_symmetric_covariance(
        canonical_axis_covariance,
        minimum_eigenvalue=minimum_covariance_eigenvalue,
    )
    return axes, canonical_axis_covariance, swapped
