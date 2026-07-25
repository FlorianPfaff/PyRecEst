"""Input-normalization helpers for hypertoroidal distributions."""

import datetime as _datetime

import numpy as np

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import all, array, isfinite

_BOOLEAN_DTYPE_NAMES = {"bool", "bool_", "torch.bool"}
_BOOLEAN_SCALAR_TYPES = (bool, np.bool_)
_COMPLEX_SCALAR_TYPES = (complex, np.complexfloating)
_TEMPORAL_SCALAR_TYPES = (
    np.datetime64,
    np.timedelta64,
    _datetime.date,
    _datetime.datetime,
    _datetime.timedelta,
)


def _reject_boolean_array(value, name: str) -> None:
    dtype = getattr(value, "dtype", None)
    if dtype is not None and str(dtype) in _BOOLEAN_DTYPE_NAMES:
        raise ValueError(f"{name} must contain real angles, not boolean values.")
    if dtype is not None and str(dtype) == "object":
        try:
            object_values = np.asarray(value, dtype=object).reshape(-1)
        except (TypeError, ValueError, RuntimeError):
            return
        if any(isinstance(item, _BOOLEAN_SCALAR_TYPES) for item in object_values):
            raise ValueError(f"{name} must contain real angles, not boolean values.")


def _reject_complex_array(value, name: str) -> None:
    if isinstance(value, _COMPLEX_SCALAR_TYPES):
        raise ValueError(f"{name} must contain real angles, not complex values.")

    dtype = getattr(value, "dtype", None)
    dtype_kind = getattr(dtype, "kind", None)
    dtype_name = "" if dtype is None else str(dtype).lower()
    if dtype_kind == "c" or "complex" in dtype_name:
        raise ValueError(f"{name} must contain real angles, not complex values.")

    if dtype is not None and dtype_kind != "O" and dtype_name != "object":
        return
    try:
        object_values = np.asarray(value, dtype=object).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return
    if any(isinstance(item, _COMPLEX_SCALAR_TYPES) for item in object_values):
        raise ValueError(f"{name} must contain real angles, not complex values.")


def _reject_temporal_array(value, name: str) -> None:
    """Reject datetime and duration values before numeric conversion."""

    if isinstance(value, _TEMPORAL_SCALAR_TYPES):
        raise ValueError(f"{name} must contain real angles, not temporal values.")

    dtype = getattr(value, "dtype", None)
    if getattr(dtype, "kind", None) in {"M", "m"}:
        raise ValueError(f"{name} must contain real angles, not temporal values.")

    try:
        object_values = np.asarray(value, dtype=object).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return
    if any(isinstance(item, _TEMPORAL_SCALAR_TYPES) for item in object_values):
        raise ValueError(f"{name} must contain real angles, not temporal values.")


def _reject_nonfinite_array(value, name: str) -> None:
    """Reject NaN and infinite angles before circular operations."""

    try:
        finite = bool(all(isfinite(value)))
    except (OverflowError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must contain only finite real angles.") from exc
    if not finite:
        raise ValueError(f"{name} must contain only finite real angles.")


def _reject_non_numeric_angle_types(value, name: str) -> None:
    """Reject values that must not be interpreted as real angles."""

    _reject_boolean_array(value, name)
    _reject_complex_array(value, name)
    _reject_temporal_array(value, name)


def _validated_real_angle_array(value, name: str):
    """Convert an input after validating its semantic and numeric domain."""

    _reject_non_numeric_angle_types(value, name)
    value = array(value)
    _reject_non_numeric_angle_types(value, name)
    _reject_nonfinite_array(value, name)
    return value


def as_shift_vector(shift_by, dim: int, *, name: str = "shift_by"):
    """Return ``shift_by`` as a one-dimensional backend vector of length ``dim``.

    A scalar shift is accepted for one-dimensional hypertoroidal distributions.
    This keeps public APIs robust for ordinary Python scalar/list inputs before
    shape validation is performed.
    """
    shift_by = _validated_real_angle_array(shift_by, name)
    if shift_by.ndim == 0:
        if dim != 1:
            raise ValueError(f"{name} must have shape ({dim},), got scalar.")
        return shift_by.reshape((1,))
    if shift_by.ndim == 1 and shift_by.shape[0] == dim:
        return shift_by
    raise ValueError(f"{name} must have shape ({dim},), got {shift_by.shape}.")


def as_hypertoroidal_points(xs, dim: int, *, name: str = "xs"):
    """Return evaluation points as an array with trailing dimension ``dim``.

    For one-dimensional distributions, a scalar is treated as one query point
    and a one-dimensional array is treated as a batch of scalar query points.
    For higher-dimensional distributions, a one-dimensional array of length
    ``dim`` is treated as a single query point.
    """
    xs = _validated_real_angle_array(xs, name)
    if xs.ndim == 0:
        if dim != 1:
            raise ValueError(f"{name} must have trailing dimension {dim}, got scalar.")
        return xs.reshape((1, 1))
    if xs.ndim == 1:
        if dim == 1:
            return xs.reshape((-1, 1))
        if xs.shape[0] == dim:
            return xs.reshape((1, dim))
        raise ValueError(f"{name} must have trailing dimension {dim}, got {xs.shape}.")
    if xs.shape[-1] != dim:
        raise ValueError(f"{name} must have trailing dimension {dim}, got {xs.shape}.")
    return xs.reshape((-1, dim))
