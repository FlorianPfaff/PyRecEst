import math as _math
import operator as _operator
import os as _os

import numpy as _np
from numpy import pi

_AXIS_FLAG_TYPES = (bool, _np.bool_)


def comb(n, k):
    return _math.comb(n, k)


def outer(a, b):
    """Return a batched outer product for array/tensor backends."""
    torch_pair = _torch_promoted_pair(a, b)
    if torch_pair is not None:
        a, b = torch_pair
        if a.ndim == 0 or b.ndim == 0:
            return a * b
        a_expanded = a[..., :, None]
        b_expanded = b[..., None, :]
        return a_expanded * b_expanded

    a = _np.asarray(a)
    b = _np.asarray(b)
    if a.ndim == 0 or b.ndim == 0:
        return _np.multiply(a, b)
    a_expanded = a[..., :, None]
    b_expanded = b[..., None, :]
    return a_expanded * b_expanded


def _normalize_size_axis(axis):
    if axis is None:
        return None
    if isinstance(axis, _AXIS_FLAG_TYPES):
        raise TypeError("an integer is required")
    try:
        return _operator.index(axis)
    except TypeError as exc:
        raise TypeError("an integer is required") from exc


def _normalize_diagonal_integer(value):
    """Return NumPy-style integer scalar arguments for stricter tensor APIs."""
    if isinstance(value, _AXIS_FLAG_TYPES):
        raise TypeError("an integer is required")
    if isinstance(value, _np.ndarray):
        if value.shape != ():
            return value
        if value.dtype.kind == "b":
            raise TypeError("an integer is required")
        if value.dtype.kind in "iu":
            return int(value.item())
        return value
    try:
        return _operator.index(value)
    except TypeError:
        return value


def size(x, axis=None):
    """Return the total number of elements or the length of a given axis."""
    axis = _normalize_size_axis(axis)
    if hasattr(x, "numel"):
        if axis is None:
            return x.numel()
        return x.shape[axis]

    shape = getattr(x, "shape", None)
    if shape is None:
        shape = _np.shape(x)

    if axis is not None:
        return shape[axis]

    result = 1
    for dim in shape:
        result *= dim
    return result


def _normalize_reduction_axes(axis, ndim_value):
    if isinstance(axis, _AXIS_FLAG_TYPES):
        raise TypeError("axis must be an integer or a sequence of integers")
    try:
        axis_index = _operator.index(axis)
    except TypeError:
        if getattr(axis, "shape", None) == ():
            raise TypeError("axis must be an integer or a sequence of integers")
        axes = tuple(axis)
        if any(isinstance(axis_index, _AXIS_FLAG_TYPES) for axis_index in axes):
            raise TypeError("axis must be an integer or a sequence of integers")
    else:
        axes = (axis_index,)

    normalized_axes = tuple(
        axis_index + ndim_value if axis_index < 0 else axis_index for axis_index in axes
    )
    if len(set(normalized_axes)) != len(normalized_axes):
        raise ValueError("duplicate value in 'axis'")

    for original_axis, normalized_axis in zip(axes, normalized_axes):
        if normalized_axis < 0 or normalized_axis >= ndim_value:
            raise IndexError(
                f"axis {original_axis} is out of bounds for array of dimension {ndim_value}"
            )

    return normalized_axes


def min(a, axis=None):  # pylint: disable=redefined-builtin
    """Return the minimum value using NumPy-style reduction axes."""
    torch = _torch_module_for_values(a)
    if torch is None:
        return _np.min(a, axis=axis)

    tensor = _torch_as_tensor_compatible(a, torch)
    if axis is None:
        return torch.min(tensor)

    result = tensor
    for one_axis in sorted(_normalize_reduction_axes(axis, tensor.ndim), reverse=True):
        result = torch.min(result, dim=one_axis).values
    return result


amin = min


def ndim(x):
    """Return the number of dimensions for arrays and array-like inputs."""
    ndim_value = getattr(x, "ndim", None)
    if ndim_value is not None:
        return ndim_value

    return _np.ndim(x)


def diagonal(a, offset=0, axis1=0, axis2=1):
    """Return selected diagonals for NumPy arrays and PyTorch tensors."""
    torch = _torch_module_for_values(a)
    if torch is not None:
        return torch.diagonal(
            _torch_as_tensor_compatible(a, torch),
            offset=_normalize_diagonal_integer(offset),
            dim1=_normalize_diagonal_integer(axis1),
            dim2=_normalize_diagonal_integer(axis2),
        )
    return _np.diagonal(a, offset=offset, axis1=axis1, axis2=axis2)


def _active_backend_name():
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except Exception:  # pragma: no cover - only during backend bootstrap/failure paths
        return _os.environ.get("PYRECEST_BACKEND")
    return getattr(backend, "__backend_name__", _os.environ.get("PYRECEST_BACKEND"))


def _torch_module_for_values(*values):
    try:
        _torch = __import__("torch")
    except ModuleNotFoundError:
        return None

    if _active_backend_name() == "pytorch" or any(
        _torch.is_tensor(value) for value in values
    ):
        return _torch
    return None


def _torch_as_tensor_compatible(value, torch, *, device=None):
    """Convert values to torch tensors, copying unsupported NumPy views first."""
    if isinstance(value, _np.ndarray) and any(stride < 0 for stride in value.strides):
        value = value.copy()
    return torch.as_tensor(value, device=device)


def _preferred_torch_device(torch, *values):
    """Return an existing non-CPU tensor device before falling back to CPU."""
    for value in values:
        if torch.is_tensor(value) and value.device.type != "cpu":
            return value.device
    for value in values:
        if torch.is_tensor(value):
            return value.device
    return None


def _torch_promoted_pair(first, second):
    torch = _torch_module_for_values(first, second)
    if torch is None:
        return None

    device = _preferred_torch_device(torch, first, second)
    first_tensor = _torch_as_tensor_compatible(first, torch, device=device)
    second_tensor = _torch_as_tensor_compatible(second, torch, device=device)
    dtype = torch.promote_types(first_tensor.dtype, second_tensor.dtype)
    return first_tensor.to(dtype=dtype), second_tensor.to(dtype=dtype)


def dot(a, b):
    """Return NumPy-compatible dot products for NumPy and PyTorch values."""
    torch_pair = _torch_promoted_pair(a, b)
    if torch_pair is not None:
        a, b = torch_pair
        torch = _torch_module_for_values(a, b)
        if a.ndim == 0 or b.ndim == 0:
            return torch.multiply(a, b)
        if a.ndim == 1 and b.ndim == 1:
            return torch.dot(a, b)
        if b.ndim == 1:
            return torch.tensordot(a, b, dims=([-1], [0]))
        if a.ndim == 1:
            return torch.tensordot(a, b, dims=([0], [-2]))
        return torch.tensordot(a, b, dims=([-1], [-2]))

    return _np.dot(_np.asarray(a), _np.asarray(b))


def matvec(matrix, vector):
    torch_pair = _torch_promoted_pair(matrix, vector)
    if torch_pair is not None:
        matrix, vector = torch_pair
        torch = _torch_module_for_values(matrix, vector)
        if vector.ndim == 1:
            return torch.matmul(matrix, vector)
        if matrix.ndim == 2:
            return torch.einsum("ij,...j->...i", matrix, vector)
        return torch.einsum("...ij,...j->...i", matrix, vector)

    matrix = _np.asarray(matrix)
    vector = _np.asarray(vector)
    if vector.ndim == 1:
        return _np.matmul(matrix, vector)
    if matrix.ndim == 2:
        return _np.einsum("ij,...j->...i", matrix, vector)
    return _np.einsum("...ij,...j->...i", matrix, vector)
