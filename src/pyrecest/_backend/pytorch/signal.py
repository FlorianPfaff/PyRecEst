import numpy as _np
import torch as _torch


def _normalize_axes(axes, ndim):
    if axes is None:
        axes = tuple(range(ndim))
    elif isinstance(axes, (int, _np.integer)):
        axes = (int(axes),)
    else:
        axes = tuple(int(axis) for axis in axes)

    normalized_axes = []
    for axis in axes:
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError(
                f"axis {axis} is out of bounds for array of dimension {ndim}"
            )
        normalized_axes.append(axis)

    if len(set(normalized_axes)) != len(normalized_axes):
        raise ValueError("duplicate value in 'axes'")
    return tuple(normalized_axes)


def _as_tensor_pair(in1, in2):
    device = next(
        (value.device for value in (in1, in2) if _torch.is_tensor(value)), None
    )
    x = in1 if _torch.is_tensor(in1) else _torch.as_tensor(in1, device=device)
    y = in2 if _torch.is_tensor(in2) else _torch.as_tensor(in2, device=x.device)
    y = y.to(device=x.device)

    dtype = _torch.promote_types(x.dtype, y.dtype)
    if not (dtype.is_floating_point or dtype.is_complex):
        dtype = _torch.get_default_dtype()
    return x.to(dtype=dtype), y.to(dtype=dtype)


def _centered_slice(full_shape, target_shape, axes):
    slices = [slice(None)] * len(full_shape)
    for axis in axes:
        start = (full_shape[axis] - target_shape[axis]) // 2
        slices[axis] = slice(start, start + target_shape[axis])
    return tuple(slices)


def _valid_slice(shape1, shape2, axes):
    slices = [slice(None)] * len(shape1)
    for axis in axes:
        start = min(shape1[axis], shape2[axis]) - 1
        length = abs(shape1[axis] - shape2[axis]) + 1
        slices[axis] = slice(start, start + length)
    return tuple(slices)


def fftconvolve(in1, in2, mode="full", axes=None):
    """Convolve two arrays using FFTs with SciPy-compatible shape semantics."""
    if mode not in {"full", "same", "valid"}:
        raise ValueError("mode must be 'full', 'same', or 'valid'")

    x, y = _as_tensor_pair(in1, in2)
    if x.ndim != y.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")

    axes = _normalize_axes(axes, x.ndim)
    if mode == "valid":
        x_larger = all(x.shape[axis] >= y.shape[axis] for axis in axes)
        y_larger = all(y.shape[axis] >= x.shape[axis] for axis in axes)
        if not (x_larger or y_larger):
            raise ValueError(
                "For mode='valid', one input must be at least as large as the other in every selected dimension."
            )

    fft_shape = tuple(int(x.shape[axis] + y.shape[axis] - 1) for axis in axes)
    real_result = not (x.dtype.is_complex or y.dtype.is_complex)

    result = _torch.fft.ifftn(
        _torch.fft.fftn(x, s=fft_shape, dim=axes)
        * _torch.fft.fftn(y, s=fft_shape, dim=axes),
        s=fft_shape,
        dim=axes,
    )
    if real_result:
        result = result.real

    if mode == "same":
        return result[_centered_slice(tuple(result.shape), tuple(x.shape), axes)]
    if mode == "valid":
        return result[_valid_slice(tuple(x.shape), tuple(y.shape), axes)]
    return result
