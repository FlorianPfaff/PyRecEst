"""Input-normalization helpers for hypertoroidal distributions."""

import numpy as np

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array

_BOOLEAN_DTYPE_NAMES = {"bool", "bool_", "torch.bool"}
_BOOLEAN_SCALAR_TYPES = (bool, np.bool_)
_COMPLEX_SCALAR_TYPES = (complex, np.complexfloating)


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


def as_shift_vector(shift_by, dim: int, *, name: str = "shift_by"):
    """Return ``shift_by`` as a one-dimensional backend vector of length ``dim``.

    A scalar shift is accepted for one-dimensional hypertoroidal distributions.
    This keeps public APIs robust for ordinary Python scalar/list inputs before
    shape validation is performed.
    """
    _reject_boolean_array(shift_by, name)
    _reject_complex_array(shift_by, name)
    shift_by = array(shift_by)
    _reject_boolean_array(shift_by, name)
    _reject_complex_array(shift_by, name)
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
    _reject_boolean_array(xs, name)
    _reject_complex_array(xs, name)
    xs = array(xs)
    _reject_boolean_array(xs, name)
    _reject_complex_array(xs, name)
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
