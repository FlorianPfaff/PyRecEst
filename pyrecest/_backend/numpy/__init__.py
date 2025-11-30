"""Numpy based computation backend."""

import numpy as _np
from numpy import (
    all,
    allclose,
    amax,
    amin,
    any,
    argmax,
    argmin,
    asarray,
    broadcast_arrays,
    broadcast_to,
    clip,
    complex64,
    complex128,
    concatenate,
    conj,
    cross,
    cumprod,
    cumsum,
    diag_indices,
    diagonal,
    einsum,
    empty_like,
    equal,
    expand_dims,
    flip,
    float32,
    float64,
    greater,
    hsplit,
    hstack,
    int32,
    int64,
    isclose,
    isnan,
    kron,
    less,
    less_equal,
    logical_and,
    logical_or,
    maximum,
    mean,
    meshgrid,
    minimum,
    moveaxis,
    ones_like,
    pad,
    prod,
    quantile,
    repeat,
    reshape,
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
    # The ones below are for pyrecest
    diag,
    diff,
    apply_along_axis,
    nonzero,
    column_stack,
    conj,
    atleast_1d,
    atleast_2d,
    dstack,
    full,
    isreal,
    triu,
    kron,
    angle,
    arctan,
    cov,
    count_nonzero,
    full_like,
    isinf,
    deg2rad,
    argsort,
    max,
    min,
    roll,
    dstack,
    round,
    array_equal,
    # For Riemannian score-based SDE
    log1p,
)

try:
    from numpy import trapezoid
except ImportError:
    from numpy import trapz as trapezoid

from scipy.special import erf, gamma, polygamma, gammaln  # NOQA

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
    dot,
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
from . import (
    autodiff,  # NOQA
    linalg,  # NOQA
    random,  # NOQA
    # For pyrecest
    fft,  # NOQA
    spatial,  # NOQA
)
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


def has_autodiff():
    """If allows for automatic differentiation.

    Returns
    -------
    has_autodiff : bool
    """
    return False


def vmap(pyfunc, randomness='error'):
    assert randomness in ('error', 'different')
    
    def vmapped_fun(*args):
        # Check if all arguments have the same first dimension
        if not all(arg.shape[0] == args[0].shape[0] for arg in args):
            raise ValueError("All arguments must have the same size in the first dimension")

        # Prepare the output array (assuming the output of pyfunc is a scalar or numpy array)
        first_output = pyfunc(*(arg[0, ...] for arg in args))
        if _np.isscalar(first_output):
            output_shape = (args[0].shape[0],)
        else:
            output_shape = (args[0].shape[0],) + first_output.shape

        output = _np.empty(output_shape)

        # Apply the function to each slice
        for i in range(args[0].shape[0]):
            output[i, ...] = pyfunc(*(arg[i, ...] for arg in args))

        return output

    return vmapped_fun