"""Numpy based computation backend."""

import builtins as _builtins

import numpy as _np
from numpy import (  # The ones below are for pyrecest; For Riemannian score-based SDE
    all,
    allclose,
    amax,
    amin,
    angle,
    any,
    apply_along_axis,
    arctan,
    argmax,
    argmin,
    argsort,
    array_equal,
    asarray,
    atleast_1d,
    atleast_2d,
    broadcast_arrays,
    broadcast_to,
    clip,
    column_stack,
    complex64,
    complex128,
    concatenate,
    conj,
    count_nonzero,
    cov,
    cross,
    cumprod,
    cumsum,
    deg2rad,
    diag,
    diag_indices,
    diagonal,
    diff,
    dot,
    dstack,
    einsum,
    empty_like,
    equal,
    expand_dims,
    flip,
    float32,
    float64,
    full,
    full_like,
    greater,
    hsplit,
    hstack,
    int32,
    int64,
    isclose,
    isfinite,
    isinf,
    isnan,
    isreal,
    isscalar,
    kron,
    less,
    less_equal,
    log1p,
    logical_and,
    logical_or,
    max,
    maximum,
    mean,
    meshgrid,
    min,
    minimum,
    moveaxis,
    nonzero,
    ones_like,
    pad,
    prod,
    quantile,
    rad2deg,
    repeat,
    reshape,
    roll,
    round,
    searchsorted,
    shape,
    sort,
    split,
    stack,
    std,
    sum,
    take,
    tile,
    transpose,
    tril,
    tril_indices,
    triu,
    triu_indices,
    uint8,
    unique,
    vstack,
    where,
    zeros_like,
)

try:
    from numpy import trapezoid
except ImportError:
    from numpy import trapz as trapezoid

from scipy.special import erf, gamma, gammaln, polygamma  # NOQA

from .._shared_numpy import (
    abs,
    angle,
    arange,
    arccos,
    arccosh,
    arcsin,
    arctan2,
    arctanh,
    array_from_sparse,
    assignment,
    assignment_by_sum,
    ceil,
    copy,
    cos,
    cosh,
    divide,
    exp,
    flatten,
    floor,
    from_numpy,
    get_slice,
    imag,
    log,
    mat_from_diag_triu_tril,
    matmul,
    matvec,
    mod,
    ndim,
    one_hot,
    outer,
    power,
    ravel_tril_indices,
    real,
    scatter_add,
    set_diag,
    sign,
    sin,
    sinh,
    sqrt,
    squeeze,
    tan,
    tanh,
    to_numpy,
    trace,
    tril_to_vec,
    triu_to_vec,
    vec_to_diag,
    vectorize,
)
from . import autodiff  # NOQA
from . import fft  # NOQA
from . import linalg  # NOQA
from . import random  # NOQA
from . import signal  # NOQA
from . import spatial  # For pyrecest; NOQA
from ._common import (
    _box_binary_scalar,
    _box_unary_scalar,
    _dyn_update_dtype,
    _modify_func_default_dtype,
    array,
    as_dtype,
    atol,
    cast,
    convert_to_wider_dtype,
    eye,
    get_default_cdtype,
    get_default_dtype,
    is_array,
    is_bool,
    is_complex,
    is_floating,
    rtol,
    set_default_dtype,
    to_ndarray,
    zeros,
)

ones = _modify_func_default_dtype(target=_np.ones)
linspace = _dyn_update_dtype(target=_np.linspace, dtype_pos=5)
empty = _dyn_update_dtype(target=_np.empty, dtype_pos=1)


def squeeze(x, axis=None):
    """Squeeze singleton axes while reporting invalid axes consistently."""

    x = _np.asarray(x)
    if axis is None:
        return _np.squeeze(x)

    if isinstance(axis, (int, _np.integer)):
        axes = (int(axis),)
    else:
        axes = tuple(axis)
    if not axes:
        return x

    normalized_axes = tuple(
        one_axis + x.ndim if one_axis < 0 else one_axis for one_axis in axes
    )
    if len(set(normalized_axes)) != len(normalized_axes):
        raise ValueError("duplicate value in 'axis'")
    for one_axis, normalized_axis in zip(axes, normalized_axes):
        if normalized_axis < 0 or normalized_axis >= x.ndim:
            raise ValueError(
                f"axis {one_axis} is out of bounds for array of dimension {x.ndim}"
            )
    if any(x.shape[one_axis] != 1 for one_axis in normalized_axes):
        return x
    squeeze_axis = normalized_axes[0] if len(normalized_axes) == 1 else normalized_axes
    return _np.squeeze(x, axis=squeeze_axis)


def trace(a, offset=0, axis1=-2, axis2=-1, dtype=None, out=None):
    """Return the trace while preserving PyRecEst's last-two-axes default."""
    return _np.trace(
        a,
        offset=offset,
        axis1=axis1,
        axis2=axis2,
        dtype=dtype,
        out=out,
    )


def has_autodiff():
    """If allows for automatic differentiation.

    Returns
    -------
    has_autodiff : bool
    """
    return False


def vmap(pyfunc, randomness="error"):
    """Vectorize ``pyfunc`` over the first axis of all positional arguments."""
    if randomness not in ("error", "different"):
        raise ValueError("randomness must be 'error' or 'different'.")

    def vmapped_fun(*args):
        if not args:
            raise ValueError("vmap requires at least one positional argument")
        mapped_args = [_np.asarray(arg) for arg in args]
        leading_sizes = [arg.shape[0] if arg.ndim > 0 else None for arg in mapped_args]
        sized_leading = [size for size in leading_sizes if size is not None]
        if sized_leading and not _builtins.all(
            size == sized_leading[0] for size in sized_leading
        ):
            raise ValueError(
                "All arguments must have the same size in the first dimension"
            )
        if _builtins.any(size is None for size in leading_sizes):
            raise ValueError("vmap arguments must have at least one dimension")

        outputs = [
            pyfunc(*(arg[index, ...] for arg in mapped_args))
            for index in range(sized_leading[0])
        ]
        return _np.stack(outputs)

    return vmapped_fun
