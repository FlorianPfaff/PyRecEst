import math as _math
import os as _os

import numpy as _np
from numpy import pi


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


def size(x, axis=None):
    """Return the total number of elements or the length of a given axis."""
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
    if isinstance(axis, (int, _np.integer)):
        axes = (int(axis),)
    else:
        axes = tuple(axis)

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

    tensor = torch.as_tensor(a)
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


def _torch_module_for_values(*values):
    try:
        import torch as _torch
    except ModuleNotFoundError:
        return None

    if _os.environ.get("PYRECEST_BACKEND") == "pytorch" or any(
        _torch.is_tensor(value) for value in values
    ):
        return _torch
    return None


def _torch_promoted_pair(first, second):
    torch = _torch_module_for_values(first, second)
    if torch is None:
        return None

    device = next(
        (value.device for value in (first, second) if torch.is_tensor(value)),
        None,
    )
    first_tensor = torch.as_tensor(first, device=device)
    second_tensor = torch.as_tensor(second, device=device)
    dtype = torch.promote_types(first_tensor.dtype, second_tensor.dtype)
    return first_tensor.to(dtype=dtype), second_tensor.to(dtype=dtype)


def _dot_b_axis(b):
    return 0 if b.ndim == 1 else -2


def dot(a, b):
    """Return NumPy-compatible dot products for NumPy and PyTorch values."""
    torch_pair = _torch_promoted_pair(a, b)
    if torch_pair is not None:
        a, b = torch_pair
        torch = _torch_module_for_values(a, b)
        if a.ndim == 0 or b.ndim == 0:
            return torch.multiply(a, b)
        return torch.tensordot(a, b, dims=([-1], [_dot_b_axis(b)]))

    a = _np.asarray(a)
    b = _np.asarray(b)
    if a.ndim == 0 or b.ndim == 0:
        return _np.multiply(a, b)
    return _np.tensordot(a, b, axes=([-1], [_dot_b_axis(b)]))


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
