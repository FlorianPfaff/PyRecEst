"""Pytorch based computation backend."""

import builtins as _builtins
from operator import index as _operator_index

import numpy as _np
import torch as _torch
from torch import (  # The ones below are for pyrecest; For Riemannian score-based SDE
    angle,
    arange,
    arctan,
    argmin,
    argsort,
    asarray,
)
from torch import broadcast_tensors as broadcast_arrays
from torch import (  # The ones below are for pyrecest; For Riemannian score-based SDE
    clip,
    column_stack,
    complex64,
    complex128,
    conj,
    count_nonzero,
    deg2rad,
    diag,
    diff,
    dstack,
    empty,
    empty_like,
    erf,
    eye,
    flatten,
    float32,
    float64,
    full,
    full_like,
    greater,
    hstack,
    int32,
    int64,
    isfinite,
    isinf,
    isnan,
    isreal,
    kron,
    less,
    log1p,
    logical_or,
    moveaxis,
    ones,
    ones_like,
    polygamma,
    quantile,
    rad2deg,
)
from torch import repeat_interleave as repeat
from torch import (  # The ones below are for pyrecest; For Riemannian score-based SDE
    reshape,
    roll,
    round,
    scatter_add,
    searchsorted,
    stack,
    trapezoid,
    triu,
    uint8,
    vmap,
    vstack,
    zeros,
    zeros_like,
)
from torch.special import gammaln
from torch.special import gammaln as _gammaln

from .._backend_config import pytorch_atol as atol
from .._backend_config import pytorch_rtol as rtol
from . import autodiff  # NOQA
from . import fft  # NOQA
from . import linalg  # NOQA
from . import random  # NOQA
from . import signal  # NOQA
from . import spatial  # for pyrecest; NOQA
from ._common import array, cast, from_numpy
from ._dtype import (
    _add_default_dtype_by_casting,
    _box_binary_scalar,
    _box_unary_scalar,
    _preserve_input_dtype,
    as_dtype,
    get_default_cdtype,
    get_default_dtype,
    is_bool,
    is_complex,
    is_floating,
    set_default_dtype,
)

_DTYPES = {
    int32: 0,
    int64: 1,
    float32: 2,
    float64: 3,
    complex64: 4,
    complex128: 5,
}


def _raise_not_implemented_error(*args, **kwargs):
    raise NotImplementedError


searchsorted = _raise_not_implemented_error


abs = _box_unary_scalar(target=_torch.abs)
angle = _box_unary_scalar(target=_torch.angle)
arccos = _box_unary_scalar(target=_torch.arccos)
arccosh = _box_unary_scalar(target=_torch.arccosh)
arcsin = _box_unary_scalar(target=_torch.arcsin)
arctanh = _box_unary_scalar(target=_torch.arctanh)
ceil = _box_unary_scalar(target=_torch.ceil)
cos = _box_unary_scalar(target=_torch.cos)
cosh = _box_unary_scalar(target=_torch.cosh)
exp = _box_unary_scalar(target=_torch.exp)
floor = _box_unary_scalar(target=_torch.floor)
log = _box_unary_scalar(target=_torch.log)
real = _box_unary_scalar(target=_torch.real)
sign = _box_unary_scalar(target=_torch.sign)
sin = _box_unary_scalar(target=_torch.sin)
sinh = _box_unary_scalar(target=_torch.sinh)
sqrt = _box_unary_scalar(target=_torch.sqrt)
tan = _box_unary_scalar(target=_torch.tan)
tanh = _box_unary_scalar(target=_torch.tanh)


arctan2 = _box_binary_scalar(target=_torch.atan2)
mod = _box_binary_scalar(target=_torch.remainder, box_x2=False)
power = _box_binary_scalar(target=_torch.pow, box_x2=False)


def _resolve_reduction_axis(axis, dim, func_name):
    if dim is not None:
        if axis is not None and axis != dim:
            raise TypeError(f"{func_name}() got both 'axis' and 'dim'")
        axis = dim
    return axis


def _resolve_keepdims(keepdims, keepdim, func_name):
    if keepdim is not None:
        if keepdims not in (False, None) and keepdims != keepdim:
            raise TypeError(f"{func_name}() got both 'keepdims' and 'keepdim'")
        keepdims = keepdim
    return keepdims


def _is_empty_reduction_axis(axis):
    if axis is None or isinstance(axis, (int, _np.integer)):
        return False
    return len(tuple(axis)) == 0


def _as_floating_reduction_input(values, dtype=None):
    values = array(values)
    if dtype is not None:
        return cast(values, dtype=dtype)
    if is_floating(values) or is_complex(values):
        return values
    return cast(values, dtype=get_default_dtype())


def _std_result_dtype(values):
    if values.dtype == complex64:
        return float32
    if values.dtype == complex128:
        return float64
    return values.dtype


def mean(a, axis=None, dtype=None, out=None, keepdims=False, *, dim=None, keepdim=None):
    axis = _resolve_reduction_axis(axis, dim, "mean")
    keepdims = _resolve_keepdims(keepdims, keepdim, "mean")
    values = _as_floating_reduction_input(a, dtype=dtype)

    if _is_empty_reduction_axis(axis):
        result = values.clone()
        if out is not None:
            out.copy_(result)
            return out
        return result

    kwargs = {"dim": axis, "keepdim": keepdims}
    if out is not None:
        kwargs["out"] = out
    return _torch.mean(values, **kwargs)


def std(
    a,
    axis=None,
    dtype=None,
    out=None,
    ddof=0,
    keepdims=False,
    *,
    correction=0,
    dim=None,
    keepdim=None,
):
    axis = _resolve_reduction_axis(axis, dim, "std")
    keepdims = _resolve_keepdims(keepdims, keepdim, "std")
    if ddof != 0 and correction != 0:
        raise ValueError("ddof and correction cannot both be nonzero")
    if correction == 0:
        correction = ddof
    a = _as_floating_reduction_input(a, dtype=dtype)

    if _is_empty_reduction_axis(axis):
        result_dtype = _std_result_dtype(a)
        if correction > 0:
            result = _torch.full_like(a, _np.nan, dtype=result_dtype)
        else:
            result = _torch.zeros_like(a, dtype=result_dtype)
        if out is not None:
            out.copy_(result)
            return out
        return result

    kwargs = {"dim": axis, "correction": correction, "keepdim": keepdims}
    if out is not None:
        kwargs["out"] = out

    return _torch.std(a, **kwargs)


def cov(input, correction=1, fweights=None, aweights=None, bias=False):
    input = array(input)
    if fweights is not None:
        fweights = asarray(fweights, device=input.device)
    if aweights is not None:
        aweights = asarray(aweights, dtype=input.dtype, device=input.device)

    if bias:
        correction = 0
    return _torch.cov(
        input, correction=correction, fweights=fweights, aweights=aweights
    )


def _quantile_q(q, x):
    if _torch.is_tensor(q):
        return q.to(device=x.device, dtype=x.dtype)
    if _np.isscalar(q):
        return float(q)
    return _torch.as_tensor(q, dtype=x.dtype, device=x.device)


def _quantile_q_shape(q):
    if _torch.is_tensor(q):
        return tuple(q.shape)
    return tuple(_np.shape(q))


def quantile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    *,
    dim=None,
    keepdim=None,
    interpolation=None,
):
    """Return quantiles using NumPy-compatible argument names."""
    del overwrite_input

    if dim is not None:
        if axis is not None and axis != dim:
            raise TypeError("quantile() got both 'axis' and 'dim'")
        axis = dim
    if keepdim is not None:
        if keepdims is not False and keepdims != keepdim:
            raise TypeError("quantile() got both 'keepdims' and 'keepdim'")
        keepdims = keepdim
    if interpolation is not None:
        method = interpolation

    x = array(a)
    if is_complex(x):
        raise TypeError("a must be an array of real numbers")
    if not is_floating(x):
        x = cast(x, dtype=get_default_dtype())

    q_arg = _quantile_q(q, x)
    q_shape = _quantile_q_shape(q)

    if axis is None or isinstance(axis, (int, _np.integer)):
        kwargs = {"dim": axis, "keepdim": keepdims, "interpolation": method}
        if out is not None:
            kwargs["out"] = out
        return _torch.quantile(x, q_arg, **kwargs)

    axes = _normalize_reduction_axes(axis, x.ndim)
    if not axes:
        result = x
        if q_shape:
            result = _torch.broadcast_to(result, q_shape + tuple(x.shape))
        if out is not None:
            out.copy_(result)
            return out
        return result

    remaining_axes = tuple(dim for dim in range(x.ndim) if dim not in axes)
    permuted = x.permute(axes + remaining_axes)
    reduced_size = int(_np.prod([x.shape[dim] for dim in axes]))
    reduced = permuted.reshape(
        (reduced_size,) + tuple(x.shape[dim] for dim in remaining_axes)
    )
    result = _torch.quantile(reduced, q_arg, dim=0, interpolation=method)

    if keepdims:
        result = result.reshape(
            q_shape + tuple(1 if dim in axes else x.shape[dim] for dim in range(x.ndim))
        )
    if out is not None:
        out.copy_(result)
        return out
    return result


def count_nonzero(a, axis=None, keepdims=False):
    """Count non-zero entries using NumPy-compatible reduction semantics."""
    x = array(a)
    if axis is None:
        result = _torch.count_nonzero(x)
        if keepdims:
            return result.reshape((1,) * x.ndim)
        return result

    counts = (x != 0).to(dtype=_torch.int64)
    return _reduce_over_axes(
        counts,
        axis,
        lambda values, one_axis, keepdim: _torch.sum(
            values, dim=one_axis, keepdim=keepdim
        ),
        keepdims,
    )


def has_autodiff():
    """If allows for automatic differentiation.

    Returns
    -------
    has_autodiff : bool
    """
    return True


def isscalar(x):
    return _np.isscalar(x)


def nonzero(x):
    """Return index arrays for non-zero elements using the NumPy contract."""
    if not _torch.is_tensor(x):
        x = _torch.as_tensor(x)
    return _torch.nonzero(x, as_tuple=True)


def matmul(x, y, out=None):
    x = array(x)
    y = array(y)
    x, y = convert_to_wider_dtype([x, y])
    return _torch.matmul(x, y, out=out)


def to_numpy(x):
    """Convert a tensor to a NumPy array without preserving autograd state."""
    if not _torch.is_tensor(x):
        return _np.asarray(x)
    return x.detach().resolve_conj().resolve_neg().cpu().numpy()


def array_equal(a, b):
    """Return whether two array-like inputs have the same shape and elements."""
    return _torch.equal(array(a), array(b))


def one_hot(labels, num_classes):
    if not _torch.is_tensor(labels):
        labels = _torch.LongTensor(labels)
    return _torch.nn.functional.one_hot(labels, num_classes).type(_torch.uint8)


def _arg_reduction(
    a,
    np_func,
    torch_func,
    func_name,
    axis=None,
    out=None,
    keepdims=False,
    *,
    dim=None,
    keepdim=None,
):
    axis = _resolve_reduction_axis(axis, dim, func_name)
    keepdims = _resolve_keepdims(keepdims, keepdim, func_name)
    a = array(a)

    if a.dtype == _torch.bool:
        result = _torch.as_tensor(
            np_func(to_numpy(a), axis=axis, keepdims=bool(keepdims)),
            device=a.device,
        )
    elif axis is None:
        result = torch_func(a)
        if keepdims:
            result = result.reshape((1,) * a.ndim)
    else:
        result = torch_func(a, dim=axis, keepdim=bool(keepdims))

    if out is not None:
        out.copy_(result)
        return out
    return result


def argmax(a, axis=None, out=None, keepdims=False, *, dim=None, keepdim=None):
    return _arg_reduction(
        a,
        _np.argmax,
        _torch.argmax,
        "argmax",
        axis,
        out,
        keepdims,
        dim=dim,
        keepdim=keepdim,
    )


def argmin(a, axis=None, out=None, keepdims=False, *, dim=None, keepdim=None):
    return _arg_reduction(
        a,
        _np.argmin,
        _torch.argmin,
        "argmin",
        axis,
        out,
        keepdims,
        dim=dim,
        keepdim=keepdim,
    )


def convert_to_wider_dtype(tensor_list):
    dtype_list = [_DTYPES.get(x.dtype, -1) for x in tensor_list]
    if len(set(dtype_list)) == 1:
        return tensor_list

    wider_dtype_index = amax(dtype_list)

    wider_dtype = list(_DTYPES.keys())[wider_dtype_index]

    tensor_list = [cast(x, dtype=wider_dtype) for x in tensor_list]
    return tensor_list


def less_equal(x, y, **kwargs):
    if not _torch.is_tensor(x):
        x = _torch.tensor(x)
    if not _torch.is_tensor(y):
        y = _torch.tensor(y)
    return _torch.le(x, y, **kwargs)


def _slice_along_axis(x, start, stop, axis):
    index = [slice(None)] * x.ndim
    index[axis] = slice(start, stop)
    return x[tuple(index)]


def split(x, indices_or_sections, axis=0):
    if not _torch.is_tensor(x):
        x = array(x)

    axis_length = x.shape[axis]
    if isinstance(indices_or_sections, (int, _np.integer)):
        n_sections = int(indices_or_sections)
        if n_sections <= 0:
            raise ValueError("number sections must be larger than 0")
        if axis_length % n_sections != 0:
            raise ValueError("array split does not result in an equal division")

        section_length = axis_length // n_sections
        return tuple(
            _slice_along_axis(
                x,
                section_index * section_length,
                (section_index + 1) * section_length,
                axis,
            )
            for section_index in range(n_sections)
        )

    cut_indices = _np.asarray(indices_or_sections)
    if cut_indices.ndim == 0:
        return split(x, int(cut_indices), axis=axis)
    if cut_indices.ndim != 1:
        raise ValueError("indices_or_sections must be a 1-D sequence")

    bounds = [None, *(int(index) for index in cut_indices.tolist()), None]
    return tuple(
        _slice_along_axis(x, start, stop, axis)
        for start, stop in zip(bounds, bounds[1:])
    )


def logical_and(x, y):
    device = None
    if _torch.is_tensor(x):
        device = x.device
    elif _torch.is_tensor(y):
        device = y.device
    return _torch.logical_and(
        _torch.as_tensor(x, device=device),
        _torch.as_tensor(y, device=device),
    )


def _normalize_reduction_axes(axis, ndim_):
    if isinstance(axis, (int, _np.integer)):
        axis = (axis,)
    else:
        axis = tuple(axis)

    normalized_axes = tuple(
        one_axis + ndim_ if one_axis < 0 else one_axis for one_axis in axis
    )
    if len(set(normalized_axes)) != len(normalized_axes):
        raise ValueError("duplicate value in 'axis'")

    for one_axis, normalized_axis in zip(axis, normalized_axes):
        if normalized_axis < 0 or normalized_axis >= ndim_:
            raise IndexError(
                f"axis {one_axis} is out of bounds for array of dimension {ndim_}"
            )

    return normalized_axes


def _reduce_over_axes(x, axis, reducer, keepdims=False):
    result = x
    for one_axis in sorted(_normalize_reduction_axes(axis, x.ndim), reverse=True):
        result = reducer(result, one_axis, bool(keepdims))
    return result


def _reduction_result(result, out=None):
    if out is not None:
        out.copy_(result)
        return out
    return result


def any(x, axis=None, out=None, keepdims=False):
    if not _torch.is_tensor(x):
        x = _torch.tensor(x)
    x = x.bool()
    if axis is None:
        result = _torch.any(x)
        if keepdims:
            result = result.reshape((1,) * x.ndim)
        return _reduction_result(result, out)
    result = _reduce_over_axes(
        x,
        axis,
        lambda values, one_axis, keepdim: _torch.any(
            values, dim=one_axis, keepdim=keepdim
        ),
        keepdims,
    )
    return _reduction_result(result, out)


def flip(x, axis):
    x = array(x)
    if isinstance(axis, int):
        axis = [axis]
    if axis is None:
        axis = list(range(x.ndim))
    return _torch.flip(x, dims=axis)


def concatenate(seq, axis=0, out=None):
    seq = _tensor_sequence(seq)
    return _torch.cat(seq, dim=axis, out=out)


def all(x, axis=None, out=None, keepdims=False):
    if not _torch.is_tensor(x):
        x = _torch.tensor(x)
    x = x.bool()
    if axis is None:
        result = _torch.all(x)
        if keepdims:
            result = result.reshape((1,) * x.ndim)
        return _reduction_result(result, out)
    result = _reduce_over_axes(
        x,
        axis,
        lambda values, one_axis, keepdim: _torch.all(
            values, dim=one_axis, keepdim=keepdim
        ),
        keepdims,
    )
    return _reduction_result(result, out)


def get_slice(x, indices):
    """Return a slice of an array, following Numpy's style.

    Parameters
    ----------
    x : array-like, shape=[dim]
        Initial array.
    indices : iterable(iterable(int))
        Indices which are kept along each axis, starting from 0.

    Returns
    -------
    slice : array-like
        Slice of x given by indices.

    Notes
    -----
    This follows Numpy's convention: indices are grouped by axis.

    Examples
    --------
    >>> a = torch.tensor(range(30)).reshape(3,10)
    >>> get_slice(a, ((0, 2), (8, 9)))
    tensor([8, 29])
    """
    return x[indices]


def allclose(a, b, atol=atol, rtol=rtol):
    if not isinstance(a, _torch.Tensor):
        a = _torch.tensor(a)
    if not isinstance(b, _torch.Tensor):
        b = _torch.tensor(b)
    a, b = convert_to_wider_dtype([a, b])
    a, b = _torch.broadcast_tensors(a, b)
    return _torch.allclose(a, b, atol=atol, rtol=rtol)


def apply_along_axis(func, axis, tensor):
    """Apply ``func`` to 1-D slices along ``axis`` with NumPy semantics."""
    if not _torch.is_tensor(tensor):
        tensor = array(tensor)

    (axis,) = _normalize_reduction_axes(axis, tensor.ndim)
    iteration_shape = tuple(tensor.shape[:axis]) + tuple(tensor.shape[axis + 1 :])

    # Move the target axis to the end so each row of ``flat_slices`` is one
    # NumPy-style 1-D slice.  The remaining dimensions keep their original
    # order and are restored below.
    moved = _torch.movedim(tensor, axis, -1)
    num_slices = int(_np.prod(iteration_shape, dtype=int)) if iteration_shape else 1
    if num_slices == 0:
        raise ValueError("Cannot apply_along_axis when any iteration dimensions are 0")
    flat_slices = moved.reshape((num_slices, moved.shape[-1]))

    output_list = []
    for tensor_slice in flat_slices:
        result_tensor = array(func(tensor_slice))
        if _torch.is_tensor(result_tensor) and result_tensor.device != tensor.device:
            result_tensor = result_tensor.to(device=tensor.device)
        output_list.append(result_tensor)

    stacked = stack(output_list, dim=0)
    result_shape = tuple(stacked.shape[1:])
    output = stacked.reshape(iteration_shape + result_shape)

    if result_shape:
        prefix_ndim = axis
        suffix_ndim = tensor.ndim - axis - 1
        result_ndim = len(result_shape)
        permutation = (
            tuple(range(prefix_ndim))
            + tuple(
                range(
                    prefix_ndim + suffix_ndim,
                    prefix_ndim + suffix_ndim + result_ndim,
                )
            )
            + tuple(range(prefix_ndim, prefix_ndim + suffix_ndim))
        )
        output = output.permute(permutation)

    return output


def shape(val):
    if not is_array(val):
        val = array(val)
    return val.shape


def max(a, axis=None, out=None, keepdims=False):
    a = array(a)
    if axis is None:
        result = _torch.max(a)
        if keepdims:
            result = result.reshape((1,) * a.ndim)
        return _reduction_result(result, out)
    result = _reduce_over_axes(
        a,
        axis,
        lambda values, one_axis, keepdim: _torch.max(
            values, dim=one_axis, keepdim=keepdim
        ).values,
        keepdims,
    )
    return _reduction_result(result, out)


amax = max


def maximum(a, b):
    return _torch.max(array(a), array(b))


def minimum(a, b):
    return _torch.min(array(a), array(b))


def to_ndarray(x, to_ndim, axis=0, dtype=None):
    x = _torch.as_tensor(x, dtype=dtype)

    if x.dim() > to_ndim:
        raise ValueError("The ndim cannot be adapted properly.")

    while x.dim() < to_ndim:
        x = _torch.unsqueeze(x, dim=axis)

    return x


def broadcast_to(x, shape):
    if not _torch.is_tensor(x):
        x = _torch.tensor(x)
    return x.expand(shape)


def isclose(x, y, rtol=rtol, atol=atol):
    if not _torch.is_tensor(x):
        x = _torch.tensor(x)
    if not _torch.is_tensor(y):
        y = _torch.tensor(y)
    x, y = convert_to_wider_dtype([x, y])
    return _torch.isclose(x, y, atol=atol, rtol=rtol)


def sum(x, axis=None, keepdims=None, dtype=None, out=None, *, dim=None, keepdim=None):
    axis = _resolve_reduction_axis(axis, dim, "sum")
    keepdims = _resolve_keepdims(keepdims, keepdim, "sum")
    x = array(x)

    if _is_empty_reduction_axis(axis):
        result = cast(x, dtype=dtype) if dtype is not None else x.clone()
        if out is not None:
            out.copy_(result)
            return out
        return result

    if axis is None:
        result = _torch.sum(x, dtype=dtype)
        if keepdims:
            result = result.reshape((1,) * x.ndim)
    elif keepdims is None:
        result = _torch.sum(x, dim=axis, dtype=dtype)
    else:
        result = _torch.sum(x, dim=axis, keepdim=keepdims, dtype=dtype)

    if out is not None:
        out.copy_(result)
        return out
    return result


def einsum(equation, *inputs):
    input_tensors_list = [arg if is_array(arg) else array(arg) for arg in inputs]
    input_tensors_list = convert_to_wider_dtype(input_tensors_list)

    return _torch.einsum(equation, *input_tensors_list)


def transpose(x, axes=None):
    if not is_array(x):
        x = array(x)
    if axes is not None:
        return x.permute(axes)
    if x.dim() == 1:
        return x
    if x.dim() > 2:
        return x.permute(tuple(range(x.ndim)[::-1]))
    return x.t()


def squeeze(x, axis=None):
    if not is_array(x):
        x = array(x)
    if axis is None:
        return _torch.squeeze(x)
    return _torch.squeeze(x, dim=axis)


def trace(x):
    if not is_array(x):
        x = array(x)
    if x.ndim == 2:
        return _torch.trace(x)

    return _torch.einsum("...ii", x)


def linspace(start, stop, num=50, endpoint=True, dtype=None):
    num = _operator_index(num)
    if num < 0:
        raise ValueError("num must be non-negative")

    device = next(
        (value.device for value in (start, stop) if _torch.is_tensor(value)),
        None,
    )

    if not _torch.is_tensor(start):
        start = _torch.as_tensor(start, dtype=dtype, device=device)
    elif dtype is not None or (device is not None and start.device != device):
        start = start.to(
            dtype=dtype if dtype is not None else start.dtype, device=device
        )

    if not _torch.is_tensor(stop):
        stop = _torch.as_tensor(stop, dtype=dtype, device=device)
    elif dtype is not None or (device is not None and stop.device != device):
        stop = stop.to(dtype=dtype if dtype is not None else stop.dtype, device=device)

    result_dtype = dtype if dtype is not None else _torch.result_type(start, stop)
    if dtype is None and not (
        result_dtype.is_floating_point or result_dtype.is_complex
    ):
        result_dtype = get_default_dtype()

    start = start.to(dtype=result_dtype)
    stop = stop.to(dtype=result_dtype)
    start, stop = _torch.broadcast_tensors(start, stop)

    fraction_dtype = result_dtype
    if result_dtype == complex64:
        fraction_dtype = float32
    elif result_dtype == complex128:
        fraction_dtype = float64
    elif not result_dtype.is_floating_point:
        fraction_dtype = get_default_dtype()

    fractions = _torch.arange(num, dtype=fraction_dtype, device=start.device)
    denominator = num - 1 if endpoint and num > 1 else num
    if denominator > 0:
        fractions = fractions / denominator
    fractions = fractions.reshape((num,) + (1,) * start.ndim)

    result = start + (stop - start) * fractions
    if result.dtype != result_dtype:
        result = result.to(dtype=result_dtype)
    return result


def equal(a, b, **kwargs):
    if not is_array(a):
        a = array(a)

    if not is_array(b):
        b = array(b)
    return _torch.eq(a, b, **kwargs)


def diag_indices(*args, **kwargs):
    return tuple(map(_torch.from_numpy, _np.diag_indices(*args, **kwargs)))


def tril(mat, k=0):
    if not is_array(mat):
        mat = array(mat)
    return _torch.tril(mat, diagonal=k)


def triu(mat, k=0):
    if not is_array(mat):
        mat = array(mat)
    return _torch.triu(mat, diagonal=k)


def tril_indices(n, k=0, m=None):
    if m is None:
        m = n
    indices = _torch.tril_indices(row=n, col=m, offset=k)
    return indices[0], indices[1]


def triu_indices(n, k=0, m=None):
    if m is None:
        m = n
    indices = _torch.triu_indices(row=n, col=m, offset=k)
    return indices[0], indices[1]


def tile(x, y):
    if not _torch.is_tensor(x):
        x = _torch.tensor(x)
    return x.repeat(y)


def atleast_1d(*arys):
    result = tuple(_torch.atleast_1d(array(ary)) for ary in arys)
    return result[0] if len(result) == 1 else result


def atleast_2d(*arys):
    result = tuple(_torch.atleast_2d(array(ary)) for ary in arys)
    return result[0] if len(result) == 1 else result


def expand_dims(x, axis=0):
    x = array(x)
    if isinstance(axis, (int, _np.integer)):
        axes = (axis,)
    else:
        axes = tuple(axis)
    output_ndim = x.ndim + len(axes)
    for one_axis in sorted(_normalize_reduction_axes(axes, output_ndim)):
        x = _torch.unsqueeze(x, dim=one_axis)
    return x


def meshgrid(*arrays, indexing="xy"):
    return _torch.meshgrid(
        *tuple(array(one_array) for one_array in arrays), indexing=indexing
    )


def ndim(x):
    if not is_array(x):
        x = array(x)
    return x.dim()


def hsplit(x, indices_or_sections):
    axis = 0 if ndim(x) == 1 else 1
    return split(x, indices_or_sections, axis=axis)


def diagonal(x, offset=0, axis1=0, axis2=1):
    if not is_array(x):
        x = array(x)
    return _torch.diagonal(x, offset=offset, dim1=axis1, dim2=axis2)


def set_diag(x, new_diag):
    """Set the diagonal along the last two axis.

    Parameters
    ----------
    x : array-like, shape=[dim]
        Initial array.
    new_diag : array-like, shape=[dim[-2]]
        Values to set on the diagonal.

    Returns
    -------
    None

    Notes
    -----
    This mimics tensorflow.linalg.set_diag(x, new_diag), when new_diag is a
    1-D array, but modifies x instead of creating a copy.
    """
    diag_len = _builtins.min(x.shape[-2], x.shape[-1])
    result = x.clone()
    diag_indices = _torch.arange(diag_len, device=x.device)
    values = _torch.as_tensor(new_diag, dtype=x.dtype, device=x.device)
    result[..., diag_indices, diag_indices] = values
    return result


def prod(x, axis=None, dtype=None, out=None, keepdims=False):
    x = array(x)
    if _is_empty_reduction_axis(axis):
        result = cast(x, dtype=dtype) if dtype is not None else x.clone()
        return _reduction_result(result, out)
    if axis is None:
        result = _torch.prod(x) if dtype is None else _torch.prod(x, dtype=dtype)
        if keepdims:
            result = result.reshape((1,) * x.ndim)
        return _reduction_result(result, out)

    def _prod_axis(values, one_axis, keepdim):
        kwargs = {"dim": one_axis, "keepdim": keepdim}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return _torch.prod(values, **kwargs)

    result = _reduce_over_axes(
        x,
        axis,
        _prod_axis,
        keepdims,
    )
    return _reduction_result(result, out)


def where(condition, x=None, y=None):
    device = next(
        (value.device for value in (x, y, condition) if _torch.is_tensor(value)),
        None,
    )
    if not _torch.is_tensor(condition):
        condition = _torch.as_tensor(condition, dtype=_torch.bool, device=device)
    else:
        condition = condition.to(device=device, dtype=_torch.bool)

    if x is None and y is None:
        return _torch.where(condition)
    if not _torch.is_tensor(x):
        x = _torch.as_tensor(x, device=device)
    elif device is not None:
        x = x.to(device=device)
    if not _torch.is_tensor(y):
        y = _torch.as_tensor(y, device=device)
    elif device is not None:
        y = y.to(device=device)
    result_dtype = _torch.result_type(x, y)
    x = x.to(dtype=result_dtype)
    y = y.to(dtype=result_dtype)
    return _torch.where(condition, x, y)


def _is_boolean(x):
    if isinstance(x, bool):
        return True
    if isinstance(x, (tuple, list)):
        if not x:
            return False
        return _is_boolean(x[0])
    if _torch.is_tensor(x):
        return x.dtype in [_torch.bool, _torch.uint8]
    return False


def _is_iterable(x):
    if isinstance(x, (list, tuple)):
        return True
    if _torch.is_tensor(x):
        return ndim(x) > 0
    return False


def _is_empty_index_sequence(indices):
    return _is_iterable(indices) and len(indices) == 0


def _as_assignment_values(values, x):
    if _torch.is_tensor(values):
        return values.to(device=x.device, dtype=x.dtype)
    return _torch.as_tensor(values, dtype=x.dtype, device=x.device)


def _assignment_value_length(values):
    return len(values) if _is_iterable(values) else 1


def _is_scalar_index(index):
    return isinstance(index, (int, _np.integer)) or (
        _torch.is_tensor(index) and index.ndim == 0
    )


def _assignment_index_length(indices, zip_indices):
    if zip_indices:
        return len(indices)
    if isinstance(indices, tuple) and _builtins.all(
        _is_scalar_index(index) for index in indices
    ):
        return 1
    return len(indices) if _is_iterable(indices) else 1


def _contains_slice(indices):
    if isinstance(indices, slice):
        return True
    if isinstance(indices, tuple):
        return _builtins.any(isinstance(index, slice) for index in indices)
    return False


def _as_assignment_index(index, *, device):
    if _torch.is_tensor(index):
        if index.dtype in [_torch.bool, _torch.uint8]:
            return index.to(device=device)
        return index.to(device=device, dtype=_torch.long)
    return _torch.as_tensor(index, dtype=_torch.long, device=device)


def _normalize_index_put_indices(indices, *, device):
    index_seq = indices if isinstance(indices, tuple) else (indices,)
    return tuple(_as_assignment_index(index, device=device) for index in index_seq)


def _as_boolean_index(indices, *, device):
    if _torch.is_tensor(indices):
        return indices.to(device=device, dtype=_torch.bool)
    return _torch.as_tensor(indices, dtype=_torch.bool, device=device)


def _apply_assignment(x_new, indices, values, *, accumulate):
    if _contains_slice(indices):
        if accumulate:
            x_new[indices] += values
        else:
            x_new[indices] = values
        return x_new
    x_new.index_put_(
        _normalize_index_put_indices(indices, device=x_new.device),
        values,
        accumulate=accumulate,
    )
    return x_new


def assignment(x, values, indices, axis=0):
    """Assign values at given indices of an array.

    Parameters
    ----------
    x: array-like, shape=[dim]
        Initial array.
    values: {float, list(float)}
        Value or list of values to be assigned.
    indices: {int, tuple, list(int), list(tuple)}
        Single int or tuple, or list of ints or tuples of indices where value
        is assigned.
        If the length of the tuples is shorter than ndim(x), values are
        assigned to each copy along axis.
    axis: int, optional
        Axis along which values are assigned, if vectorized.

    Returns
    -------
    x_new : array-like, shape=[dim]
        Copy of x with the values assigned at the given indices.

    Notes
    -----
    If a single value is provided, it is assigned at all the indices.
    If a list is given, it must have the same length as indices.
    """
    x_new = copy(array(x))
    if _is_empty_index_sequence(indices):
        return x_new

    values = _as_assignment_values(values, x_new)
    use_vectorization = hasattr(indices, "__len__") and len(indices) < ndim(x_new)
    if _is_boolean(indices):
        indices = _as_boolean_index(indices, device=x_new.device)
        x_new[indices] = values
        return x_new
    zip_indices = (
        _is_iterable(indices) and len(indices) > 0 and _is_iterable(indices[0])
    )
    len_indices = _assignment_index_length(indices, zip_indices)
    if zip_indices:
        indices = tuple(zip(*indices))
    if not use_vectorization:
        len_values = _assignment_value_length(values)
        if (
            not _contains_slice(indices)
            and len_values > 1
            and len_values != len_indices
        ):
            raise ValueError("Either one value or as many values as indices")
        _apply_assignment(x_new, indices, values, accumulate=False)
    else:
        indices = tuple(list(indices[:axis]) + [slice(None)] + list(indices[axis:]))
        x_new[indices] = values
    return x_new


def assignment_by_sum(x, values, indices, axis=0):
    """Add values at given indices of an array.

    Parameters
    ----------
    x: array-like, shape=[dim]
        Initial array.
    values: {float, list(float)}
        Value or list of values to be assigned.
    indices: {int, tuple, list(int), list(tuple)}
        Single int or tuple, or list of ints or tuples of indices where value
        is assigned.
        If the length of the tuples is shorter than ndim(x), values are
        assigned to each copy along axis.
    axis: int, optional
        Axis along which values are assigned, if vectorized.

    Returns
    -------
    x_new : array-like, shape=[dim]
        Copy of x with the values assigned at the given indices.

    Notes
    -----
    If a single value is provided, it is assigned at all the indices.
    If a list is given, it must have the same length as indices.
    """
    x_new = copy(array(x))
    if _is_empty_index_sequence(indices):
        return x_new

    values = _as_assignment_values(values, x_new)
    use_vectorization = hasattr(indices, "__len__") and len(indices) < ndim(x_new)
    if _is_boolean(indices):
        indices = _as_boolean_index(indices, device=x_new.device)
        x_new[indices] += values
        return x_new
    zip_indices = (
        _is_iterable(indices) and len(indices) > 0 and _is_iterable(indices[0])
    )
    len_indices = _assignment_index_length(indices, zip_indices)
    if zip_indices:
        indices = tuple(zip(*indices))
    if not use_vectorization:
        len_values = _assignment_value_length(values)
        if (
            not _contains_slice(indices)
            and len_values > 1
            and len_values != len_indices
        ):
            raise ValueError("Either one value or as many values as indices")
        _apply_assignment(x_new, indices, values, accumulate=True)
    else:
        indices = tuple(list(indices[:axis]) + [slice(None)] + list(indices[axis:]))
        x_new[indices] += values
    return x_new


def copy(x):
    if _torch.is_tensor(x):
        return x.clone()
    return _np.copy(x)


def cumsum(x, axis=None, dtype=None):
    if not _torch.is_tensor(x):
        x = array(x, dtype=dtype)
    if axis is None:
        return x.flatten().cumsum(dim=0, dtype=dtype)
    return _torch.cumsum(x, dim=axis, dtype=dtype)


def cumprod(x, axis=None, dtype=None):
    if not _torch.is_tensor(x):
        x = array(x, dtype=dtype)
    if axis is None:
        return _torch.cumprod(x.flatten(), dim=0, dtype=dtype)
    return _torch.cumprod(x, dim=axis, dtype=dtype)


def array_from_sparse(indices, data, target_shape):
    """Create an array of given shape, with values at specific indices.

    The rest of the array will be filled with zeros.

    Parameters
    ----------
    indices : iterable(tuple(int))
        Index of each element which will be assigned a specific value.
    data : iterable(scalar)
        Value associated at each index.
    target_shape : tuple(int)
        Shape of the output array.

    Returns
    -------
    a : array, shape=target_shape
        Array of zeros with specified values assigned to specified indices.
    """
    data = array(data)
    indices = _torch.as_tensor(indices, dtype=_torch.long, device=data.device)
    if indices.numel() == 0:
        if data.numel() != 0:
            raise ValueError("data must be empty when indices are empty")
        return _torch.zeros(
            _torch.Size(target_shape), dtype=data.dtype, device=data.device
        )

    return _torch.sparse_coo_tensor(
        indices.t(),
        data,
        _torch.Size(target_shape),
        device=data.device,
    ).to_dense()


def vectorize(x, pyfunc, multiple_args=False, **kwargs):
    if multiple_args:
        return stack(list(map(lambda y: pyfunc(*y), zip(*x))))
    return stack(list(map(pyfunc, x)))


def _tensor_sequence(seq):
    tensors = [array(item) for item in seq]
    if not tensors:
        return tensors
    return convert_to_wider_dtype(tensors)


def stack(seq, axis=0, out=None, *, dim=None):
    if dim is not None:
        if axis not in (0, dim):
            raise TypeError("stack() got both 'axis' and 'dim'")
        axis = dim

    return _torch.stack(_tensor_sequence(seq), dim=axis, out=out)


def vec_to_diag(vec):
    return _torch.diag_embed(vec, offset=0)


def tril_to_vec(x, k=0):
    n = x.shape[-1]
    rows, cols = tril_indices(n, k=k)
    return x[..., rows, cols]


def triu_to_vec(x, k=0):
    n = x.shape[-1]
    rows, cols = triu_indices(n, k=k)
    return x[..., rows, cols]


def mat_from_diag_triu_tril(diag, tri_upp, tri_low):
    """Build matrix from given components.

    Forms a matrix from diagonal, strictly upper triangular and
    strictly lower traingular parts.

    Parameters
    ----------
    diag : array_like, shape=[..., n]
    tri_upp : array_like, shape=[..., (n * (n - 1)) / 2]
    tri_low : array_like, shape=[..., (n * (n - 1)) / 2]

    Returns
    -------
    mat : array_like, shape=[..., n, n]
    """
    diag = array(diag)
    tri_upp = array(tri_upp)
    tri_low = array(tri_low)
    diag, tri_upp, tri_low = convert_to_wider_dtype([diag, tri_upp, tri_low])

    n = diag.shape[-1]
    (i,) = diag_indices(n, ndim=1)
    j, k = triu_indices(n, k=1)
    i = i.to(device=diag.device)
    j = j.to(device=diag.device)
    k = k.to(device=diag.device)
    mat = _torch.zeros((diag.shape + (n,)), dtype=diag.dtype, device=diag.device)
    mat[..., i, i] = diag
    mat[..., j, k] = tri_upp
    mat[..., k, j] = tri_low
    return mat


def divide(a, b, ignore_div_zero=False):
    a = array(a)
    b = array(b)
    non_cpu_device = next(
        (value.device for value in (a, b) if value.device.type != "cpu"),
        None,
    )
    if non_cpu_device is not None:
        a = a.to(device=non_cpu_device)
        b = b.to(device=non_cpu_device)
    a, b = convert_to_wider_dtype([a, b])
    quotient = _torch.divide(a, b)

    if ignore_div_zero is False:
        return quotient

    zero = _torch.zeros((), dtype=quotient.dtype, device=quotient.device)
    return _torch.where(b != 0, quotient, zero)


def ravel_tril_indices(n, k=0, m=None):
    if m is None:
        size = (n, n)
    else:
        size = (n, m)
    idxs = _np.tril_indices(n, k, m)
    return _torch.from_numpy(_np.ravel_multi_index(idxs, size))


def sort(a, axis=-1):
    a = array(a)
    sorted_a, _ = _torch.sort(a, dim=axis)
    return sorted_a


def min(a, axis=None, out=None, keepdims=False):
    a = array(a)
    if axis is None:
        result = _torch.min(a)
        if keepdims:
            result = result.reshape((1,) * a.ndim)
        return _reduction_result(result, out)
    result = _reduce_over_axes(
        a,
        axis,
        lambda values, one_axis, keepdim: _torch.min(
            values, dim=one_axis, keepdim=keepdim
        ).values,
        keepdims,
    )
    return _reduction_result(result, out)


amin = min


def _normalize_take_axis(axis, ndim_):
    if axis is None:
        return None
    axis = int(axis)
    if axis < 0:
        axis += ndim_
    if axis < 0 or axis >= ndim_:
        raise IndexError(f"axis {axis} is out of bounds for array of dimension {ndim_}")
    return axis


def _normalize_take_indices(indices, axis_size, mode):
    if mode is None:
        mode = "raise"
    if mode == "raise":
        if _torch.any((indices >= axis_size) | (indices < -axis_size)):
            raise IndexError("index out of bounds")
        return _torch.remainder(indices, axis_size)
    if mode == "wrap":
        if axis_size == 0 and indices.numel():
            raise IndexError("cannot do a non-empty take from an empty axis")
        return _torch.remainder(indices, axis_size) if axis_size else indices
    if mode == "clip":
        if axis_size == 0 and indices.numel():
            raise IndexError("cannot do a non-empty take from an empty axis")
        return _torch.clamp(indices, 0, axis_size - 1) if axis_size else indices
    raise ValueError("mode must be one of 'raise', 'wrap', or 'clip'")


def take(a, indices, axis=None, out=None, mode=None):
    a = array(a)
    axis = _normalize_take_axis(axis, a.ndim)
    if axis is None:
        a = _torch.flatten(a)
        axis = 0

    if not _torch.is_tensor(indices):
        indices = _torch.as_tensor(indices, dtype=_torch.long, device=a.device)
    else:
        indices = indices.to(device=a.device, dtype=_torch.long)

    scalar_index = indices.ndim == 0
    indices_shape = tuple(indices.shape)
    flat_indices = indices.reshape(-1)
    flat_indices = _normalize_take_indices(flat_indices, a.shape[axis], mode)

    result = _torch.index_select(a, axis, flat_indices)
    if scalar_index:
        result = _torch.squeeze(result, dim=axis)
    else:
        result = result.reshape(
            tuple(a.shape[:axis]) + indices_shape + tuple(a.shape[axis + 1 :])
        )

    if out is not None:
        out.copy_(result)
        return out
    return result


def _torch_pad_width(pad_width, ndim_):
    try:
        pad_pairs = _np.broadcast_to(_np.asarray(pad_width), (ndim_, 2))
    except ValueError as exc:
        raise ValueError(
            f"pad_width must be broadcastable to shape ({ndim_}, 2)"
        ) from exc

    if _np.any(pad_pairs < 0):
        raise ValueError("index can't contain negative values")

    return [int(value) for pair in reversed(pad_pairs.tolist()) for value in pair]


def _trim_leading_zero_pad_pairs_for_nonconstant_mode(torch_pad_width, ndim_):
    """Drop zero-padded leading axes unsupported by PyTorch nonconstant padding."""

    # ``torch.nn.functional.pad`` supports nonconstant modes only for a suffix
    # of dimensions.  PyRecEst accepts NumPy-style per-axis pad widths, so a
    # common image-shaped request such as ``((0, 0), (0, 0), (1, 1), (1, 1))``
    # becomes ``[1, 1, 1, 1, 0, 0, 0, 0]`` for PyTorch.  The trailing zero pairs
    # address leading axes and must be stripped before calling PyTorch.
    min_supported_pairs = {
        2: 1,
        3: 1,
        4: 2,
        5: 3,
    }.get(ndim_)
    if min_supported_pairs is None:
        return torch_pad_width

    pad_pairs = [
        torch_pad_width[index : index + 2]
        for index in range(0, len(torch_pad_width), 2)
    ]
    while len(pad_pairs) > min_supported_pairs and pad_pairs[-1] == [0, 0]:
        pad_pairs.pop()

    return [value for pair in pad_pairs for value in pair]


def pad(a, pad_width, mode="constant", constant_values=0.0):
    a = array(a)
    torch_pad_width = _torch_pad_width(pad_width, a.ndim)
    if mode != "constant":
        torch_pad_width = _trim_leading_zero_pad_pairs_for_nonconstant_mode(
            torch_pad_width,
            a.ndim,
        )
        return _torch.nn.functional.pad(a, torch_pad_width, mode=mode)

    return _torch.nn.functional.pad(
        a, torch_pad_width, mode=mode, value=constant_values
    )


def is_array(x):
    return _torch.is_tensor(x)


def outer(a, b):
    a = array(a)
    b = array(b)
    a, b = convert_to_wider_dtype([a, b])

    # TODO: improve for torch > 1.9 (dims=0 fails in 1.9)
    if a.ndim == 0 or b.ndim == 0:
        return _torch.multiply(a, b)
    return _torch.einsum("...i,...j->...ij", a, b)


def matvec(A, b):
    A = array(A)
    b = array(b)
    A, b = convert_to_wider_dtype([A, b])

    if A.ndim == 2 and b.ndim == 1:
        return _torch.mv(A, b)

    if b.ndim == 1:  # A.ndim > 2
        return _torch.matmul(A, b)

    if A.ndim == 2:  # b.ndim > 1
        return _torch.einsum("ij,...j->...i", A, b)

    return _torch.einsum("...ij,...j->...i", A, b)


def dot(a, b):
    a = array(a)
    b = array(b)
    a, b = convert_to_wider_dtype([a, b])

    if a.ndim == 0 or b.ndim == 0:
        return _torch.multiply(a, b)

    if a.ndim == 1 and b.ndim == 1:
        return _torch.dot(a, b)

    if b.ndim == 1:
        return _torch.einsum("...i,i->...", a, b)

    if a.ndim == 1:
        return _torch.einsum("i,...i->...", a, b)

    return _torch.einsum("...i,...i->...", a, b)


def _normalize_cross_axis(axis, ndim_, name):
    axis = _operator_index(axis)
    if axis < 0:
        axis += ndim_
    if axis < 0 or axis >= ndim_:
        raise IndexError(
            f"{name} {axis} is out of bounds for array of dimension {ndim_}"
        )
    return axis


def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    if axis is not None:
        axisa = axis
        axisb = axis
        axisc = axis

    a = array(a)
    b = array(b)
    a, b = convert_to_wider_dtype([a, b])

    axisa = _normalize_cross_axis(axisa, a.ndim, "axisa")
    axisb = _normalize_cross_axis(axisb, b.ndim, "axisb")
    a = _torch.movedim(a, axisa, -1)
    b = _torch.movedim(b, axisb, -1)

    a_dim = a.shape[-1]
    b_dim = b.shape[-1]
    if a_dim not in (2, 3) or b_dim not in (2, 3):
        raise ValueError(
            "incompatible dimensions for cross product " "(dimension must be 2 or 3)"
        )

    leading_shape = _np.broadcast_shapes(tuple(a.shape[:-1]), tuple(b.shape[:-1]))
    if tuple(a.shape[:-1]) != leading_shape:
        a = _torch.broadcast_to(a, leading_shape + (a_dim,))
    if tuple(b.shape[:-1]) != leading_shape:
        b = _torch.broadcast_to(b, leading_shape + (b_dim,))

    if a_dim == 2 and b_dim == 2:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    if a_dim == 3 and b_dim == 3:
        result = _torch.cross(a, b, dim=-1)
    elif a_dim == 2:
        result = stack(
            [
                a[..., 1] * b[..., 2],
                -a[..., 0] * b[..., 2],
                a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
            ],
            dim=-1,
        )
    else:
        result = stack(
            [
                -a[..., 2] * b[..., 1],
                a[..., 2] * b[..., 0],
                a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
            ],
            dim=-1,
        )

    axisc = _normalize_cross_axis(axisc, result.ndim, "axisc")
    return _torch.movedim(result, -1, axisc)


def gamma(a):
    return _torch.exp(_gammaln(a))


def imag(a):
    if not _torch.is_tensor(a):
        a = _torch.tensor(a)
    if is_complex(a):
        return _torch.imag(a)
    return _torch.zeros(a.shape, dtype=a.dtype, device=a.device)


def unique(ar, axis=None):
    return _torch.unique(array(ar), dim=axis)
