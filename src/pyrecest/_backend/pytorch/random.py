"""Torch based random backend."""

from math import prod as _prod
from numbers import Integral as _Integral

import torch as _torch
from torch import get_rng_state as get_state  # For PyRecEst
from torch import set_rng_state as set_state  # For PyRecEst
from torch.distributions.multivariate_normal import (
    MultivariateNormal as _MultivariateNormal,
)

from ._dtype import _allow_complex_dtype, _modify_func_default_dtype

_COMPLEX_TO_FLOAT_DTYPE = {
    _torch.complex64: _torch.float32,
    _torch.complex128: _torch.float64,
}


def _size_type_error():
    return TypeError("size must be None, an integer, or a sequence of integers")


def _looks_like_integer_dimension(value):
    return isinstance(value, _Integral) and not isinstance(value, bool)


def _integer_dimension(value):
    if not _looks_like_integer_dimension(value):
        raise _size_type_error()
    value = int(value)
    if value < 0:
        raise ValueError("size dimensions must be non-negative")
    return value


def _shape_from_size(size):
    if size is None:
        return ()
    if _looks_like_integer_dimension(size):
        return (_integer_dimension(size),)
    if isinstance(size, (str, bytes)) or not hasattr(size, "__iter__"):
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


def _randint_array(low, high, size, *args, **kwargs):
    if args:
        raise TypeError(
            "array-valued randint bounds do not support additional positional arguments"
        )

    dtype = kwargs.pop("dtype", None)
    if dtype is None:
        dtype = _torch.int64
    device = kwargs.pop("device", None)
    generator = kwargs.pop("generator", None)
    out = kwargs.pop("out", None)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

    device = _randint_device(low, high, device=device)
    low = _torch.as_tensor(low, device=device)
    high = _torch.as_tensor(high, device=device)
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


def _normal_device(*values):
    for value in values:
        if _torch.is_tensor(value):
            return value.device
    return None


def _normal_array_parameters(loc, scale):
    device = _normal_device(loc, scale)
    loc = _torch.as_tensor(loc, device=device)
    scale = _torch.as_tensor(scale, device=device)
    dtype = _torch.promote_types(
        _torch.result_type(loc, scale),
        _torch.get_default_dtype(),
    )
    return loc.to(dtype=dtype), scale.to(dtype=dtype)


def _integer_population_size(a):
    if isinstance(a, _Integral):
        return int(a)
    if (
        _torch.is_tensor(a)
        and a.ndim == 0
        and not _torch.is_floating_point(a)
        and not _torch.is_complex(a)
    ):
        return int(a.item())
    return None


def _choice_indices(
    population_size, size, num_samples, replace, p, device, *, shuffle=True
):
    if population_size <= 0:
        if num_samples == 0:
            return _torch.empty(size or (0,), dtype=_torch.long, device=device)
        raise ValueError("a must be greater than 0 unless no samples are taken")

    if p is not None:
        if not replace and num_samples > population_size:
            raise ValueError(
                "Cannot take a larger sample than population when 'replace=False'."
            )

        p = _torch.as_tensor(p, dtype=_torch.float32, device=device)
        if p.ndim != 1 or p.shape[0] != population_size:
            raise ValueError(
                "p must be 1-dimensional with one entry per population item"
            )

        p_sum = p.sum()
        if (
            bool(_torch.any(p < 0))
            or not bool(_torch.isfinite(p_sum))
            or bool(p_sum <= 0)
        ):
            raise ValueError("probabilities do not sum to a positive value")
        p = p / p_sum
        indices = _torch.multinomial(p, num_samples=num_samples, replacement=replace)
        if size is None:
            return indices[0]
        return indices.reshape(size)

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
    return _torch.rand(_shape_from_rand_args(dims, size), dtype=dtype)


def _multinomial_sample_count(sample_shape):
    return _prod(sample_shape) if sample_shape else 1


def multinomial(n, pvals, size=None):
    if not _looks_like_integer_dimension(n):
        raise TypeError("n must be a non-negative integer")
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")

    sample_shape = _shape_from_size(size)
    device = pvals.device if _torch.is_tensor(pvals) else None
    pvals = _torch.as_tensor(pvals, dtype=_torch.float32, device=device)
    if pvals.ndim != 1:
        raise ValueError("pvals must be 1-dimensional")
    if pvals.numel() == 0:
        raise ValueError("pvals must contain at least one probability")

    p_sum = pvals.sum()
    if (
        bool(_torch.any(pvals < 0))
        or not bool(_torch.isfinite(p_sum))
        or bool(p_sum <= 0)
    ):
        raise ValueError("probabilities do not sum to a positive value")
    pvals = pvals / p_sum

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
        return _torch.normal(mean=loc, std=scale, size=size or ())

    loc, scale = _normal_array_parameters(loc, scale)
    if size is None:
        return _torch.normal(mean=loc, std=scale)

    if bool(_torch.any(scale < 0)):
        raise ValueError("scale must be non-negative")
    dtype = _torch.result_type(loc, scale)
    return _torch.empty(size, dtype=dtype, device=loc.device).normal_() * scale + loc


def _uniform_size(size, low, high):
    if size is not None:
        return _shape_from_size(size)

    try:
        return tuple(_torch.broadcast_shapes(low.shape, high.shape))
    except RuntimeError as exc:
        raise ValueError("low and high could not be broadcast together") from exc


def uniform(low=0.0, high=1.0, size=None, dtype=None):
    device = None
    if _torch.is_tensor(low):
        device = low.device
    elif _torch.is_tensor(high):
        device = high.device

    low = _torch.as_tensor(low, dtype=dtype, device=device)
    high = _torch.as_tensor(high, dtype=dtype, device=device)
    size = _uniform_size(size, low, high)
    if bool(_torch.any(low > high)):
        raise ValueError("Upper bound must be greater than or equal to lower bound")
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


@_modify_func_default_dtype(copy=False, kw_only=True)
@_allow_complex_dtype
def multivariate_normal(mean, cov, size=None):
    device = _tensor_device(mean, cov)
    dtype = _floating_distribution_dtype(mean, cov)
    mean = _torch.as_tensor(mean, dtype=dtype, device=device)
    cov = _torch.as_tensor(cov, dtype=mean.dtype, device=mean.device)
    return _MultivariateNormal(mean, cov).sample(_normal_sample_size(size))
