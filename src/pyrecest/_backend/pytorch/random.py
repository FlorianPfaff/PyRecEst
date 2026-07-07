"""Torch based random backend."""

from math import prod as _prod
from numbers import Integral as _Integral

import numpy as _np
import torch as _torch
from torch import get_rng_state as get_state  # For PyRecEst
from torch import set_rng_state as set_state  # For PyRecEst
from torch.distributions.multivariate_normal import (
    MultivariateNormal as _MultivariateNormal,
)

from ._dtype import (
    _allow_complex_dtype,
    _modify_func_default_dtype,
    _normalize_torch_dtype,
)

_COMPLEX_TO_FLOAT_DTYPE = {
    _torch.complex64: _torch.float32,
    _torch.complex128: _torch.float64,
}
_BOOLEAN_TYPES = (bool, _np.bool_)
_INTEGER_DTYPES = {
    _torch.uint8,
    _torch.int8,
    _torch.int16,
    _torch.int32,
    _torch.int64,
}
_TORCH_DTYPE_BY_NAME = {
    "bool": _torch.bool,
    "uint8": _torch.uint8,
    "int8": _torch.int8,
    "int16": _torch.int16,
    "int32": _torch.int32,
    "int64": _torch.int64,
    "float16": _torch.float16,
    "float32": _torch.float32,
    "float64": _torch.float64,
    "complex64": _torch.complex64,
    "complex128": _torch.complex128,
}


def _size_type_error():
    return TypeError("size must be None, an integer, or a sequence of integers")


def _scalar_integer_dimension(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, _Integral):
        return int(value)
    if isinstance(value, _np.ndarray) and value.ndim == 0:
        if _np.issubdtype(value.dtype, _np.integer) and not _np.issubdtype(
            value.dtype, _np.bool_
        ):
            return int(value.item())
        return None
    if _torch.is_tensor(value) and value.ndim == 0 and value.dtype in _INTEGER_DTYPES:
        return int(value.item())
    return None


def _looks_like_integer_dimension(value):
    return _scalar_integer_dimension(value) is not None


def _is_zero_dimensional_array_like(value):
    return (isinstance(value, _np.ndarray) and value.ndim == 0) or (
        _torch.is_tensor(value) and value.ndim == 0
    )


def _integer_dimension(value):
    value = _scalar_integer_dimension(value)
    if value is None:
        raise _size_type_error()
    if value < 0:
        raise ValueError("size dimensions must be non-negative")
    return value


def _shape_from_size(size):
    if size is None:
        return ()
    if _looks_like_integer_dimension(size):
        return (_integer_dimension(size),)
    if (
        isinstance(size, (str, bytes))
        or _is_zero_dimensional_array_like(size)
        or not hasattr(size, "__iter__")
    ):
        raise _size_type_error()
    return tuple(_integer_dimension(dim) for dim in size)


def _shape_from_rand_args(dims, size):
    """Convert NumPy-style ``rand`` dimensions and ``size=`` to a shape."""
    if dims:
        if size is not None:
            raise TypeError("Specify either positional dimensions or size, not both.")
        size = dims[0] if len(dims) == 1 else dims
    return _shape_from_size(size)


def _choice_size(size):
    if size is None:
        return None, 1
    size = _shape_from_size(size)
    return size, _prod(size) if size else 1


def _contains_boolean_value(value):
    if isinstance(value, _BOOLEAN_TYPES):
        return True
    if _torch.is_tensor(value):
        return value.dtype == _torch.bool
    try:
        values = _np.asarray(value, dtype=object).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return False
    return any(
        isinstance(item, _BOOLEAN_TYPES)
        or (_torch.is_tensor(item) and item.ndim == 0 and item.dtype == _torch.bool)
        for item in values
    )


def _choice_bool(value, name):
    if isinstance(value, _BOOLEAN_TYPES):
        return bool(value)
    if _torch.is_tensor(value) and value.ndim == 0 and value.dtype == _torch.bool:
        return bool(value.item())
    if isinstance(value, _np.ndarray) and value.shape == () and value.dtype.kind == "b":
        return bool(value.item())
    raise TypeError(f"{name} must be a boolean")


def _is_real_numeric_dtype(dtype):
    return dtype.is_floating_point or dtype in _INTEGER_DTYPES


def _coerce_probability_tensor(value, name, device):
    if _contains_boolean_value(value):
        raise TypeError(f"{name} must be real numeric, not boolean")
    try:
        if _torch.is_tensor(value):
            values = value.to(device=device) if device is not None else value
        else:
            value_array = _np.asarray(value)
            if value_array.dtype.kind not in "iuf":
                raise TypeError
            values = _torch.as_tensor(value_array, dtype=_torch.float64, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"{name} must be real numeric") from exc
    if not _is_real_numeric_dtype(values.dtype):
        raise TypeError(f"{name} must be real numeric")
    return values.to(dtype=_torch.float64)


def _normalize_probability_tensor(values):
    if values.numel() == 0:
        scale = values.new_tensor(0.0)
    else:
        scale = values.max()
    if (
        bool(_torch.any(values < 0))
        or not bool(_torch.isfinite(scale))
        or bool(scale <= 0)
    ):
        raise ValueError("probabilities do not sum to a positive value")
    scaled = values / scale
    total = scaled.sum()
    if not bool(_torch.isfinite(total)) or bool(total <= 0):
        raise ValueError("probabilities do not sum to a positive value")
    return scaled / total


def _validate_choice_probabilities(p, population_size, device):
    p = _coerce_probability_tensor(p, "p", device)
    if p.ndim != 1 or p.shape[0] != population_size:
        raise ValueError("p must be 1-dimensional with one entry per population item")
    return _normalize_probability_tensor(p)


def _randint_size(size):
    return _shape_from_size(size)


def _is_array_parameter(value):
    return (
        _torch.is_tensor(value)
        or isinstance(value, (list, tuple))
        or (not isinstance(value, (str, bytes)) and hasattr(value, "__array__"))
    )


def _randint_device(*values, device=None):
    if device is not None:
        return device
    for value in values:
        if _torch.is_tensor(value):
            return value.device
    return None


def _randint_array_size(size, low, high):
    try:
        parameter_shape = tuple(_torch.broadcast_shapes(low.shape, high.shape))
    except RuntimeError as exc:
        raise ValueError("low and high could not be broadcast together") from exc
    if size is None:
        return parameter_shape
    sample_shape = _shape_from_size(size)
    try:
        broadcast_shape = tuple(_torch.broadcast_shapes(sample_shape, parameter_shape))
    except RuntimeError as exc:
        raise ValueError("size, low, and high could not be broadcast together") from exc
    if broadcast_shape != sample_shape:
        raise ValueError("size, low, and high could not be broadcast together")
    return sample_shape


def _validate_randint_array_bound(name, bound):
    if (
        bound.dtype == _torch.bool
        or bound.dtype.is_floating_point
        or bound.dtype.is_complex
    ):
        raise TypeError(f"{name} must contain integer values")


def _normalize_random_dtype(dtype, *, default):
    dtype = _normalize_torch_dtype(dtype, default=default)
    if dtype is None or isinstance(dtype, _torch.dtype):
        return dtype
    try:
        return _TORCH_DTYPE_BY_NAME[str(_np.dtype(dtype))]
    except (KeyError, TypeError):
        return dtype


def _normalize_randint_dtype(dtype):
    """Return an integer dtype for randint outputs."""
    dtype = _normalize_random_dtype(dtype, default=_torch.int64)
    if dtype not in _INTEGER_DTYPES:
        raise TypeError("dtype must be an integer dtype")
    return dtype


def _normalize_torch_dtype_kwargs(kwargs):
    if "dtype" not in kwargs:
        return kwargs
    kwargs = dict(kwargs)
    kwargs["dtype"] = _normalize_random_dtype(kwargs["dtype"], default=None)
    return kwargs


def _randint_array(low, high, size, *args, **kwargs):
    if args:
        raise TypeError(
            "array-valued randint bounds do not support additional positional arguments"
        )
    dtype = _normalize_randint_dtype(kwargs.pop("dtype", None))
    device = kwargs.pop("device", None)
    generator = kwargs.pop("generator", None)
    out = kwargs.pop("out", None)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

    device = _randint_device(low, high, device=device)
    low = _torch.as_tensor(low, device=device)
    high = _torch.as_tensor(high, device=device)
    _validate_randint_array_bound("low", low)
    _validate_randint_array_bound("high", high)
    sample_shape = _randint_array_size(size, low, high)
    try:
        low = _torch.broadcast_to(low, sample_shape)
        high = _torch.broadcast_to(high, sample_shape)
    except RuntimeError as exc:
        raise ValueError("size, low, and high could not be broadcast together") from exc
    if bool(_torch.any(high <= low)):
        raise ValueError("high must be greater than low")

    span = high - low
    unit_samples = _torch.rand(sample_shape, device=device, generator=generator)
    result = _torch.floor(unit_samples * span).to(dtype=dtype) + low.to(dtype=dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result


def randint(low, high=None, size=None, *args, **kwargs):
    kwargs = _normalize_torch_dtype_kwargs(kwargs)
    if high is None:
        if low is None:
            raise TypeError("randint() missing required argument 'high'")
        if _is_array_parameter(low):
            return _randint_array(0, low, size, *args, **kwargs)
        return _torch.randint(low, _randint_size(size), *args, **kwargs)
    if _is_array_parameter(low) or _is_array_parameter(high):
        return _randint_array(low, high, size, *args, **kwargs)
    return _torch.randint(low, high, _randint_size(size), *args, **kwargs)


def _normal_size(size):
    if size is None:
        return None
    return _shape_from_size(size)


def _broadcasted_parameter_shape(*parameters, message):
    try:
        return tuple(
            _torch.broadcast_shapes(*(parameter.shape for parameter in parameters))
        )
    except RuntimeError as exc:
        raise ValueError(message) from exc


def _sample_shape_from_size_and_parameters(size, parameters, message):
    parameter_shape = _broadcasted_parameter_shape(*parameters, message=message)
    if size is None:
        return parameter_shape
    sample_shape = _shape_from_size(size)
    try:
        broadcast_shape = tuple(_torch.broadcast_shapes(sample_shape, parameter_shape))
    except RuntimeError as exc:
        raise ValueError(message) from exc
    if broadcast_shape != sample_shape:
        raise ValueError(message)
    return sample_shape


def _normal_array_size(size, loc, scale):
    return _sample_shape_from_size_and_parameters(
        size,
        (loc, scale),
        "size, loc, and scale could not be broadcast together",
    )


def _normal_device(*values):
    for value in values:
        if _torch.is_tensor(value):
            return value.device
    return None


def _validate_normal_parameter(value, name, *, device=None):
    if _contains_boolean_value(value):
        raise TypeError(f"{name} must be real numeric, not boolean")
    try:
        parameter = _torch.as_tensor(value, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"{name} must be real numeric") from exc
    if not _is_real_numeric_dtype(parameter.dtype):
        raise TypeError(f"{name} must be real numeric")
    if bool(_torch.any(~_torch.isfinite(parameter))):
        raise ValueError(f"{name} must be finite")
    return parameter


def _validate_normal_scale(scale, *, device=None):
    scale = _validate_normal_parameter(scale, "scale", device=device)
    if bool(_torch.any(scale < 0)):
        raise ValueError("scale must be non-negative")
    return scale


def _normal_array_parameters(loc, scale):
    device = _normal_device(loc, scale)
    loc = _validate_normal_parameter(loc, "loc", device=device)
    scale = _validate_normal_scale(scale, device=device)
    dtype = _torch.promote_types(
        _torch.result_type(loc, scale),
        _torch.get_default_dtype(),
    )
    return loc.to(dtype=dtype), scale.to(dtype=dtype)


def _integer_population_size(a):
    if isinstance(a, bool):
        return None
    if isinstance(a, _Integral):
        return int(a)
    if isinstance(a, _np.ndarray) and a.ndim == 0:
        if _np.issubdtype(a.dtype, _np.integer) and not _np.issubdtype(
            a.dtype, _np.bool_
        ):
            return int(a.item())
        return None
    if (
        _torch.is_tensor(a)
        and a.ndim == 0
        and a.dtype != _torch.bool
        and not _torch.is_floating_point(a)
        and not _torch.is_complex(a)
    ):
        return int(a.item())
    return None


def _empty_choice_indices(size, device):
    return _torch.empty(size or (0,), dtype=_torch.long, device=device)


def _choice_indices(
    population_size, size, num_samples, replace, p, device, *, shuffle=True
):
    if population_size <= 0:
        if num_samples == 0:
            return _empty_choice_indices(size, device)
        raise ValueError("a must be greater than 0 unless no samples are taken")

    if p is not None:
        if not replace and num_samples > population_size:
            raise ValueError(
                "Cannot take a larger sample than population when 'replace=False'."
            )
        p = _validate_choice_probabilities(p, population_size, device)
        if num_samples == 0:
            return _empty_choice_indices(size, p.device)
        indices = _torch.multinomial(p, num_samples=num_samples, replacement=replace)
        if size is None:
            return indices[0]
        return indices.reshape(size)

    if num_samples == 0:
        return _empty_choice_indices(size, device)

    if replace:
        return _torch.randint(0, population_size, size or (), device=device)

    if num_samples > population_size:
        raise ValueError(
            "Cannot take a larger sample than population when 'replace=False'."
        )
    indices = _torch.randperm(population_size, device=device)[:num_samples]
    if not bool(shuffle):
        indices = _torch.sort(indices).values
    if size is None:
        return indices[0]
    return indices.reshape(size)


def _normalize_axis(axis, ndim):
    if not _looks_like_integer_dimension(axis):
        raise TypeError("axis must be an integer")
    axis = int(axis)
    if axis < -ndim or axis >= ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    return axis % ndim


def _take_choice(a, indices, axis):
    axis = _normalize_axis(axis, a.ndim)
    if indices.ndim == 0:
        return a.select(axis, int(indices.item()))
    flattened_indices = indices.reshape(-1)
    selected = _torch.index_select(a, dim=axis, index=flattened_indices)
    return selected.reshape((*a.shape[:axis], *indices.shape, *a.shape[axis + 1 :]))


def choice(a, size=None, replace=True, p=None, axis=0, shuffle=True):
    replace = _choice_bool(replace, "replace")
    shuffle = _choice_bool(shuffle, "shuffle")
    size, num_samples = _choice_size(size)
    population_size = _integer_population_size(a)
    if population_size is not None:
        device = p.device if _torch.is_tensor(p) else None
        return _choice_indices(
            population_size, size, num_samples, replace, p, device, shuffle=shuffle
        )

    if not _torch.is_tensor(a):
        a = _torch.as_tensor(a)
    if a.ndim == 0:
        raise ValueError(
            "a must be a positive integer or an array with at least one dimension"
        )
    axis = _normalize_axis(axis, a.ndim)
    indices = _choice_indices(
        a.shape[axis], size, num_samples, replace, p, a.device, shuffle=shuffle
    )
    return _take_choice(a, indices, axis)


def seed(*args, **kwargs):
    return _torch.manual_seed(*args, **kwargs)


def rand(*dims, size=None, dtype=None):
    dtype = _normalize_random_dtype(dtype, default=None)
    return _torch.rand(_shape_from_rand_args(dims, size), dtype=dtype)


def _multinomial_sample_count(sample_shape):
    return _prod(sample_shape) if sample_shape else 1


def _validate_multinomial_pvals(pvals, device):
    return _coerce_probability_tensor(pvals, "pvals", device)


def multinomial(n, pvals, size=None):
    if not _looks_like_integer_dimension(n):
        raise TypeError("n must be a non-negative integer")
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")

    sample_shape = _shape_from_size(size)
    device = pvals.device if _torch.is_tensor(pvals) else None
    pvals = _validate_multinomial_pvals(pvals, device)
    if pvals.ndim != 1:
        raise ValueError("pvals must be 1-dimensional")
    if pvals.numel() == 0:
        raise ValueError("pvals must contain at least one probability")

    pvals = _normalize_probability_tensor(pvals)

    output_shape = (*sample_shape, pvals.shape[0])
    sample_count = _multinomial_sample_count(sample_shape)
    if n == 0 or sample_count == 0:
        return _torch.zeros(output_shape, dtype=_torch.long, device=pvals.device)

    samples = _torch.multinomial(pvals.expand(sample_count, -1), n, replacement=True)
    counts = _torch.nn.functional.one_hot(samples, num_classes=pvals.shape[0]).sum(
        dim=-2
    )
    return counts.reshape(output_shape)


@_allow_complex_dtype
def normal(loc=0.0, scale=1.0, size=None):
    size = _normal_size(size)
    if not (_is_array_parameter(loc) or _is_array_parameter(scale)):
        _validate_normal_parameter(loc, "loc")
        _validate_normal_scale(scale)
        return _torch.normal(mean=loc, std=scale, size=size or ())

    loc, scale = _normal_array_parameters(loc, scale)
    size = _normal_array_size(size, loc, scale)
    dtype = _torch.result_type(loc, scale)
    return _torch.empty(size, dtype=dtype, device=loc.device).normal_() * scale + loc


def _uniform_size(size, low, high):
    return _sample_shape_from_size_and_parameters(
        size,
        (low, high),
        "size, low, and high could not be broadcast together",
    )


def _validate_uniform_bound(bound, name, *, dtype=None, device=None):
    if _contains_boolean_value(bound):
        raise TypeError(f"{name} must be real numeric, not boolean")
    try:
        bound = _torch.as_tensor(bound, dtype=dtype, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"{name} must be real numeric") from exc
    if not _is_real_numeric_dtype(bound.dtype):
        raise TypeError(f"{name} must be real numeric")
    if bool(_torch.any(~_torch.isfinite(bound))):
        raise ValueError("uniform bounds must be finite")
    return bound


def _validate_uniform_bounds(low, high):
    if bool(_torch.any(low > high)):
        raise ValueError("Upper bound must be greater than or equal to lower bound")


def uniform(low=0.0, high=1.0, size=None, dtype=None):
    dtype = _normalize_random_dtype(dtype, default=None)
    device = None
    if _torch.is_tensor(low):
        device = low.device
    elif _torch.is_tensor(high):
        device = high.device
    low = _validate_uniform_bound(low, "low", dtype=dtype, device=device)
    high = _validate_uniform_bound(high, "high", dtype=dtype, device=device)
    size = _uniform_size(size, low, high)
    _validate_uniform_bounds(low, high)
    return (high - low) * _torch.rand(size, dtype=dtype, device=device) + low


def _tensor_device(*values):
    for value in values:
        if _torch.is_tensor(value):
            return value.device
    return None


def _floating_distribution_dtype(*values):
    for value in values:
        if not _torch.is_tensor(value):
            continue
        if value.dtype.is_floating_point:
            return value.dtype
        if value.dtype.is_complex:
            return _COMPLEX_TO_FLOAT_DTYPE[value.dtype]
    return _torch.get_default_dtype()


def _normal_sample_size(size):
    return _shape_from_size(size)


def _validate_multivariate_normal_parameter(value, name, *, dtype, device):
    if _contains_boolean_value(value):
        raise TypeError(f"{name} must be real numeric, not boolean")
    try:
        parameter = _torch.as_tensor(value, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"{name} must be real numeric") from exc
    if not _is_real_numeric_dtype(parameter.dtype):
        raise TypeError(f"{name} must be real numeric")
    if bool(_torch.any(~_torch.isfinite(parameter))):
        raise ValueError(f"{name} must be finite")
    return parameter.to(dtype=dtype)


@_modify_func_default_dtype(copy=False, kw_only=True)
@_allow_complex_dtype
def multivariate_normal(mean, cov, size=None):
    device = _tensor_device(mean, cov)
    dtype = _floating_distribution_dtype(mean, cov)
    mean = _validate_multivariate_normal_parameter(
        mean,
        "mean",
        dtype=dtype,
        device=device,
    )
    cov = _validate_multivariate_normal_parameter(
        cov,
        "cov",
        dtype=mean.dtype,
        device=mean.device,
    )
    return _MultivariateNormal(mean, cov).sample(_normal_sample_size(size))
