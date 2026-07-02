import numpy as _np
import torch as _torch

_AXIS_TYPE_ERROR = "axes must be None, an integer, or a sequence of integers"


def _coerce_axis(axis):
    try:
        axis_array = _np.asarray(axis)
    except (TypeError, ValueError) as exc:
        raise TypeError(_AXIS_TYPE_ERROR) from exc
    if axis_array.shape != ():
        raise TypeError(_AXIS_TYPE_ERROR)
    try:
        return int(axis_array.item().__index__())
    except AttributeError as exc:
        raise TypeError(_AXIS_TYPE_ERROR) from exc


def _is_scalar_axis_argument(value):
    if isinstance(value, (int, _np.integer, bool, _np.bool_)):
        return True
    return getattr(value, "shape", None) == ()


def _normalize_axes(axes, ndim):
    axes_were_provided = axes is not None
    if axes is None:
        axes = tuple(range(ndim))
    elif _is_scalar_axis_argument(axes):
        axes = (_coerce_axis(axes),)
    else:
        try:
            axes = tuple(_coerce_axis(axis) for axis in axes)
        except TypeError as exc:
            raise TypeError(_AXIS_TYPE_ERROR) from exc

    if axes_were_provided and len(axes) == 0 and ndim > 0:
        raise ValueError("when provided, axes cannot be empty")

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


def _validate_non_convolved_shapes(shape1, shape2, axes):
    axes_set = set(axes)
    for axis, (first, second) in enumerate(zip(shape1, shape2, strict=True)):
        if axis in axes_set or first == second or first == 1 or second == 1:
            continue
        raise ValueError(f"incompatible shapes for in1 and in2: {shape1} and {shape2}")


def _as_torch_compatible(value):
    if isinstance(value, _np.ndarray) and any(stride < 0 for stride in value.strides):
        return value.copy()
    return value


def _preferred_tensor_device(*values):
    tensor_devices = [value.device for value in values if _torch.is_tensor(value)]
    if not tensor_devices:
        return None
    return next(
        (device for device in tensor_devices if device.type != "cpu"),
        tensor_devices[0],
    )


def _as_tensor_pair(in1, in2):
    device = _preferred_tensor_device(in1, in2)
    x = (
        in1
        if _torch.is_tensor(in1)
        else _torch.as_tensor(_as_torch_compatible(in1), device=device)
    )
    if device is not None:
        x = x.to(device=device)
    y = (
        in2
        if _torch.is_tensor(in2)
        else _torch.as_tensor(_as_torch_compatible(in2), device=x.device)
    )
    y = y.to(device=x.device)

    dtype = _torch.promote_types(x.dtype, y.dtype)
    if not (dtype.is_floating_point or dtype.is_complex):
        dtype = _torch.get_default_dtype()
    return x.to(dtype=dtype), y.to(dtype=dtype)


def _centered_slice(full_shape, target_shape):
    slices = []
    for full, target in zip(full_shape, target_shape, strict=True):
        start = (full - target) // 2
        slices.append(slice(start, start + target))
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
    x_shape = tuple(x.shape)
    y_shape = tuple(y.shape)
    _validate_non_convolved_shapes(x_shape, y_shape, axes)
    if mode == "valid":
        comparison_axes = tuple(
            axis for axis in axes if x.shape[axis] != 1 and y.shape[axis] != 1
        )
        x_larger = all(x.shape[axis] >= y.shape[axis] for axis in comparison_axes)
        y_larger = all(y.shape[axis] >= x.shape[axis] for axis in comparison_axes)
        if not (x_larger or y_larger):
            raise ValueError(
                "For mode='valid', one input must be at least as large as the other in every selected non-singleton dimension."
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
        return result[_centered_slice(tuple(result.shape), x_shape)]
    if mode == "valid":
        return result[_valid_slice(x_shape, y_shape, axes)]
    return result
