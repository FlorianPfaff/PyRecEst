"""Autograd based computation backend."""

import autograd.numpy as _np
from autograd.numpy import (
    all,
    allclose,
    amax,
    amin,
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
    from autograd.numpy import trapezoid
except ImportError:
    from autograd.numpy import trapz as trapezoid

from autograd.scipy.special import erf, gamma, gammaln, polygamma  # NOQA

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
    cos,
    cosh,
    divide,
    dot,
    exp,
    flatten,
    floor,
    from_numpy,
    get_slice,
    log,
    mat_from_diag_triu_tril,
    matmul,
    matvec,
    mod,
    ndim,
    one_hot,
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
from . import spatial  # NOQA
from ._common import (
    _box_binary_scalar,
    _box_unary_scalar,
    _dyn_update_dtype,
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

ones = _dyn_update_dtype(target=_np.ones)
linspace = _dyn_update_dtype(target=_np.linspace)
empty = _dyn_update_dtype(target=_np.empty)


def has_autodiff():
    """If allows for automatic differentiation.

    Returns
    -------
    has_autodiff : bool
    """
    return True


def vmap(pyfunc, randomness="error"):
    """Vectorize ``pyfunc`` over the first axis of all positional arguments."""
    if randomness not in ("error", "different"):
        raise ValueError("randomness must be either 'error' or 'different'")

    def vmapped_fun(*args):
        if not all([arg.shape[0] == args[0].shape[0] for arg in args]):
            raise ValueError(
                "All arguments must have the same size in the first dimension"
            )

        first_output = pyfunc(*(arg[0, ...] for arg in args))
        if _np.isscalar(first_output):
            output_shape = (args[0].shape[0],)
        else:
            output_shape = (args[0].shape[0],) + first_output.shape

        output = empty(output_shape)
        for i in range(args[0].shape[0]):
            output[i, ...] = pyfunc(*(arg[i, ...] for arg in args))
        return output

    return vmapped_fun


def imag(x):
    out = _np.imag(x)
    if is_array(x):
        return out

    return array(out)


def copy(x):
    return _np.array(x, copy=True)


def outer(a, b):
    if a.ndim > 1 and b.ndim > 1:
        return _np.einsum("...i,...j->...ij", a, b)

    out = _np.outer(a, b).reshape(a.shape + b.shape)
    if b.ndim > 1:
        out = out.swapaxes(0, -2)

    return out
