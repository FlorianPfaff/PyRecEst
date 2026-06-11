"""Jax-based computation backend.
based on implementation by Emile Mathieu
for Riemannian Score-based SDE
"""

import builtins as _builtins
import numbers as _numbers

import jax.numpy as _jnp
from jax import vmap
from jax.numpy import (  # For pyrecest; For Riemannian score-based SDE
    abs,
    all,
    allclose,
    amax,
    amin,
    angle,
    any,
    apply_along_axis,
    arange,
    arccos,
    arccosh,
    arcsin,
    arctan,
    arctan2,
    arctanh,
    argmax,
    argmin,
    argsort,
    array_equal,
    asarray,
    atleast_1d,
    atleast_2d,
    broadcast_arrays,
    broadcast_to,
    ceil,
    clip,
    column_stack,
    complex64,
    complex128,
    concatenate,
    conj,
    copy,
    cos,
    cosh,
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
    divide,
    dot,
    dstack,
    einsum,
    empty,
    empty_like,
    equal,
    exp,
    expand_dims,
    eye,
    flip,
    float32,
    float64,
    floor,
    full,
    full_like,
    greater,
    hsplit,
    hstack,
    imag,
    int32,
    int64,
    isclose,
    isfinite,
    isinf,
    isnan,
    isreal,
    kron,
    less,
    less_equal,
    linspace,
    log,
    log1p,
    logical_and,
    logical_or,
    matmul,
    max,
    maximum,
    mean,
    min,
    minimum,
    mod,
    moveaxis,
    ndim,
    nonzero,
    ones,
    ones_like,
    outer,
    pad,
    power,
    prod,
    quantile,
    rad2deg,
    real,
    repeat,
    reshape,
    roll,
    round,
    searchsorted,
    shape,
    sign,
    sin,
    sinh,
    sort,
    split,
    sqrt,
    squeeze,
    stack,
    std,
    sum,
    tan,
    tanh,
    tile,
    transpose,
    tril,
    tril_indices,
    triu,
    triu_indices,
    uint8,
    unique,
    vectorize,
    vstack,
    where,
    zeros,
    zeros_like,
)


def has_autodiff():
    """If allows for automatic differentiation.

    Returns
    -------
    has_autodiff : bool
    """
    return True


def isscalar(x):
    return _jnp.isscalar(x) and not isinstance(x, _jnp.ndarray)


def meshgrid(*xi, copy=True, sparse=False, indexing="xy"):
    """Return coordinate matrices with NumPy-style axis coercion.

    The PyRecEst backend contract follows NumPy semantics: callers may pass
    lists, ranges, and scalar axes. JAX's native ``meshgrid`` requires array or
    scalar arguments and then rejects 0-D array axes. Coercing every axis to at
    least one-dimensional JAX arrays preserves NumPy-compatible behavior while
    keeping the returned arrays in the JAX backend.
    """
    axes = tuple(_jnp.atleast_1d(_jnp.asarray(axis)) for axis in xi)
    return _jnp.meshgrid(*axes, copy=copy, sparse=sparse, indexing=indexing)


from jax import device_get as to_numpy
from jax.numpy import array
from jax.numpy import asarray as from_numpy
from jax.numpy import ravel as flatten
from jax.scipy.integrate import trapezoid
from jax.scipy.integrate import trapezoid as trapz
from jax.scipy.special import erf, gamma, gammaln, polygamma

from .._backend_config import jax_atol as atol
from .._backend_config import jax_rtol as rtol
from . import fft  # For PyRecEst
from . import signal  # For PyRecEst
from . import spatial  # For PyRecEst
from . import autodiff, linalg, random
from ._dtype import as_dtype, set_default_dtype


def _asarray_sequence(seq):
    return tuple(_jnp.asarray(item) for item in seq)


def concatenate(seq, axis=0, dtype=None):
    return _jnp.concatenate(_asarray_sequence(seq), axis=axis, dtype=dtype)


def stack(seq, axis=0, dtype=None):
    return _jnp.stack(_asarray_sequence(seq), axis=axis, dtype=dtype)


def flip(m, axis=None):
    return _jnp.flip(_jnp.asarray(m), axis=axis)


def sort(a, axis=-1, **kwargs):
    return _jnp.sort(_jnp.asarray(a), axis=axis, **kwargs)


def unique(ar, *args, **kwargs):
    return _jnp.unique(_jnp.asarray(ar), *args, **kwargs)


def _asarray_or_none(value):
    return None if value is None else _jnp.asarray(value)


def cov(
    m,
    y=None,
    rowvar=True,
    bias=False,
    ddof=None,
    fweights=None,
    aweights=None,
    dtype=None,
):
    return _jnp.cov(
        _jnp.asarray(m),
        y=_asarray_or_none(y),
        rowvar=rowvar,
        bias=bias,
        ddof=ddof,
        fweights=_asarray_or_none(fweights),
        aweights=_asarray_or_none(aweights),
        dtype=dtype,
    )


def diagonal(a, offset=0, axis1=0, axis2=1):
    return _jnp.diagonal(_jnp.asarray(a), offset=offset, axis1=axis1, axis2=axis2)


def squeeze(a, axis=None):
    return _jnp.squeeze(_jnp.asarray(a), axis=axis)


def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    return _jnp.trace(
        _jnp.asarray(a),
        offset=offset,
        axis1=axis1,
        axis2=axis2,
        dtype=dtype,
        out=out,
    )


def tril(m, k=0):
    return _jnp.tril(_jnp.asarray(m), k=k)


def triu(m, k=0):
    return _jnp.triu(_jnp.asarray(m), k=k)


def argmax(a, axis=None, out=None, keepdims=False, **kwargs):
    result = _jnp.argmax(_jnp.asarray(a), axis=axis, keepdims=keepdims, **kwargs)
    if out is not None:
        return out.at[...].set(result)
    return result


def argmin(a, axis=None, out=None, keepdims=False, **kwargs):
    result = _jnp.argmin(_jnp.asarray(a), axis=axis, keepdims=keepdims, **kwargs)
    if out is not None:
        return out.at[...].set(result)
    return result


def convert_to_wider_dtype(*args, **kwargs):
    raise NotImplementedError(
        "The function convert_to_wider_dtype is not supported in this JAX backend."
    )


def get_default_dtype(*args, **kwargs):
    raise NotImplementedError(
        "The function get_default_dtype is not supported in this JAX backend."
    )


def get_default_cdtype(*args, **kwargs):
    raise NotImplementedError(
        "The function get_default_cdtype is not supported in this JAX backend."
    )


def to_ndarray(x, to_ndim, axis=0):
    """
    Convert an input to a JAX array and adjust its dimensionality if necessary.

    Parameters
    ----------
    x : array-like or scalar
        Input data, which could be a list, tuple, scalar, or an existing JAX array.
    to_ndim : int
        Target number of dimensions for the output array.
    axis : int, optional
        The axis along which a new dimension should be inserted, if needed.

    Returns
    -------
    x : jax.numpy.ndarray
        A JAX array with the desired number of dimensions.
    """
    # Ensure the input is a JAX array
    if not isinstance(x, _jnp.ndarray):
        x = _jnp.array(x)

    if x.ndim > to_ndim:
        raise ValueError("The ndim cannot be adapted properly.")

    while x.ndim < to_ndim:
        x = _jnp.expand_dims(x, axis=axis)

    return x


def take(
    a,
    indices,
    axis=None,
    out=None,
    mode=None,
    unique_indices=False,
    indices_are_sorted=False,
    fill_value=None,
):
    return _jnp.take(
        a,
        _jnp.asarray(indices),
        axis=axis,
        out=out,
        mode=mode,
        unique_indices=unique_indices,
        indices_are_sorted=indices_are_sorted,
        fill_value=fill_value,
    )


def _is_boolean_index(indices):
    if isinstance(indices, (bool, _jnp.bool_)):
        return True
    if isinstance(indices, (list, tuple)):
        return bool(indices) and _is_boolean_index(indices[0])
    if isinstance(indices, _jnp.ndarray):
        return indices.dtype in (_jnp.bool_, _jnp.uint8)
    return False


def _is_iterable_index(indices):
    if isinstance(indices, (list, tuple)):
        return True
    if isinstance(indices, _jnp.ndarray):
        return indices.ndim > 0
    return False


def _is_scalar_index(index):
    return isinstance(index, _numbers.Integral) or (
        isinstance(index, _jnp.ndarray) and index.ndim == 0
    )


def _assignment_value_length(values):
    if isinstance(values, (list, tuple)):
        return len(values)
    if isinstance(values, _jnp.ndarray) and values.ndim > 0:
        return values.shape[0]
    return 1


def _normalize_assignment_index(indices, ndim_x, axis=0):
    if _is_boolean_index(indices):
        return _jnp.asarray(indices), False, None

    use_vectorization = _is_iterable_index(indices) and len(indices) < ndim_x
    zip_indices = (
        _is_iterable_index(indices)
        and len(indices) > 0
        and _is_iterable_index(indices[0])
    )

    if use_vectorization:
        normalized = tuple(list(indices[:axis]) + [slice(None)] + list(indices[axis:]))
        return normalized, True, None

    if zip_indices:
        normalized = tuple(_jnp.asarray(index_axis) for index_axis in zip(*indices))
        return normalized, False, len(indices)

    if isinstance(indices, list):
        return _jnp.asarray(indices), False, len(indices)
    if isinstance(indices, _jnp.ndarray) and indices.ndim > 0:
        return indices, False, indices.shape[0]
    if isinstance(indices, tuple):
        if _builtins.all(_is_scalar_index(index) for index in indices):
            return indices, False, 1
        if indices and _is_iterable_index(indices[0]):
            return indices, False, len(indices[0])
    return indices, False, 1


def _validate_assignment_value_count(values, *, use_vectorization, len_indices):
    if use_vectorization or len_indices is None:
        return
    len_values = _assignment_value_length(values)
    if len_values > 1 and len_values != len_indices:
        raise ValueError("Either one value or as many values as indices required")


def assignment(x, values, indices, axis=0):
    """
    Assign values at given indices of an array using JAX.

    Parameters
    ----------
    x: JAX array, shape=[dim]
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
    x_new : JAX array, shape=[dim]
        Copy of x with the values assigned at the given indices.
    """
    x = _jnp.asarray(x)
    normalized_indices, use_vectorization, len_indices = _normalize_assignment_index(
        indices,
        x.ndim,
        axis=axis,
    )
    _validate_assignment_value_count(
        values,
        use_vectorization=use_vectorization,
        len_indices=len_indices,
    )
    return x.at[normalized_indices].set(values)


def assignment_by_sum(x, values, indices, axis=0):
    """
    Add values at given indices of a JAX array.

    Parameters
    ----------
    x : JAX array, shape=[dim]
        Initial array.
    values : {float, list(float)}
        Value or list of values to be added.
    indices : {int, tuple, list(int), list(tuple)}
        Single int or tuple, or list of ints or tuples of indices where value is added.
        If the length of the tuples is shorter than ndim(x), values are
        assigned to each copy along axis.
    axis: int, optional
        Axis along which values are assigned, if vectorized.

    Returns
    -------
    x_new : JAX array, shape=[dim]
        Copy of x with the values added at the given indices.

    Notes
    -----
    If a single value is provided, it is added at all the indices.
    If a list is given, it must have the same length as indices.
    """
    x = _jnp.asarray(x)
    normalized_indices, use_vectorization, len_indices = _normalize_assignment_index(
        indices,
        x.ndim,
        axis=axis,
    )
    _validate_assignment_value_count(
        values,
        use_vectorization=use_vectorization,
        len_indices=len_indices,
    )
    return x.at[normalized_indices].add(values)


def array_from_sparse(indices, data, target_shape):
    """
    Create an array of given shape, with values at specific indices.
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
    # Convert inputs to JAX arrays if they aren't already
    indices = _jnp.array(indices)
    data = _jnp.array(data)

    # Create a dense array of zeros with the appropriate data type
    out = _jnp.zeros(target_shape, dtype=data.dtype)

    # Compute linear indices from multi-dimensional indices and apply them to
    # the flattened dense array.  Indexing the original n-D array with these
    # flattened positions would address the first axis instead of the flat
    # storage order and can therefore put values into the wrong entries or
    # raise out-of-bounds errors for multidimensional target shapes.
    linear_indices = _jnp.ravel_multi_index(indices.T, target_shape)
    out = out.reshape(-1).at[linear_indices].set(data).reshape(target_shape)

    return out


def is_complex(x):
    return _jnp.iscomplexobj(x)


def cast(array, dtype):
    return _jnp.asarray(array, dtype=dtype)


def ravel_tril_indices(n, k=0, m=None):
    if m is None:
        m = n
    rows, cols = _jnp.tril_indices(n, k=k, m=m)
    return _jnp.ravel_multi_index((rows, cols), (n, m))


def is_array(obj):
    return isinstance(obj, _jnp.ndarray)


def get_slice(array, indices):
    return array[indices]


# Check if dtype is floating-point
def is_floating(array):
    return _jnp.issubdtype(array.dtype, _jnp.floating)


# Check if dtype is boolean
def is_bool(array):
    return _jnp.issubdtype(array.dtype, _jnp.bool_)


def divide(a, b, ignore_div_zero=False):
    a_arr, b_arr = _jnp.asarray(a), _jnp.asarray(b)
    if ignore_div_zero is False:
        return _jnp.divide(a_arr, b_arr)

    quotient = _jnp.divide(a_arr, b_arr)
    return _jnp.where(b_arr != 0, quotient, _jnp.zeros_like(quotient))


def dot(a, b):
    a = _jnp.asarray(a)
    b = _jnp.asarray(b)

    if a.ndim == 0 or b.ndim == 0:
        return _jnp.multiply(a, b)
    if b.ndim == 1:
        return _jnp.einsum("...i,i->...", a, b)
    if a.ndim == 1:
        return _jnp.einsum("i,...i->...", a, b)
    return _jnp.einsum("...i,...i->...", a, b)


def matmul(x, y, out=None):
    x = _jnp.asarray(x)
    y = _jnp.asarray(y)
    result = _jnp.matmul(x, y)
    if out is None:
        return result
    return result


def outer(a, b):
    a = _jnp.asarray(a)
    b = _jnp.asarray(b)

    if a.ndim > 1 and b.ndim > 1:
        return _jnp.einsum("...i,...j->...ij", a, b)
    if a.ndim == 1 and b.ndim > 1:
        return _jnp.einsum("i,...j->...ij", a, b)
    if a.ndim > 1 and b.ndim == 1:
        return _jnp.einsum("...i,j->...ij", a, b)
    return _jnp.multiply.outer(a, b)


# Matrix-vector multiplication
def matvec(matrix, vector):
    matrix = _jnp.asarray(matrix)
    vector = _jnp.asarray(vector)

    if vector.ndim == 1:
        return _jnp.matmul(matrix, vector)
    if matrix.ndim == 2:
        return _jnp.einsum("ij,...j->...i", matrix, vector)
    return _jnp.einsum("...ij,...j->...i", matrix, vector)


def trace(a):
    return _jnp.trace(_jnp.asarray(a), axis1=-2, axis2=-1)


# One-hot encoding
def one_hot(indices, depth):
    return _jnp.eye(depth, dtype=_jnp.uint8)[_jnp.asarray(indices)]


# Scatter-add operation
def scatter_add(input, dim, index, src):
    """Add ``src`` into ``input`` at ``index`` along ``dim``.

    This mirrors the shared backend contract used by the facade instead of
    JAX's lower-level ``.at`` update signature. Existing input values are
    preserved, matching NumPy/PyTorch scatter-add behavior.
    """
    input = _jnp.asarray(input)
    index = _jnp.asarray(index)
    src = _jnp.asarray(src, dtype=input.dtype)

    if dim < 0:
        dim += input.ndim

    if dim == 0:
        return input.at[index].add(src)

    if dim == 1:
        if input.ndim < 2:
            raise ValueError("dim=1 scatter_add requires at least two dimensions")
        if index.ndim == 1:
            row_indices = _jnp.arange(input.shape[0])
        else:
            row_shape = (input.shape[0],) + (1,) * (index.ndim - 1)
            row_indices = _jnp.broadcast_to(
                _jnp.arange(input.shape[0]).reshape(row_shape), index.shape
            )
        return input.at[row_indices, index].add(src)

    raise NotImplementedError("scatter_add is implemented for dim 0 and dim 1.")


def set_diag(matrix, values):
    matrix = _jnp.asarray(matrix)
    values = _jnp.asarray(values, dtype=matrix.dtype)
    diag_len = _builtins.min(matrix.shape[-2], matrix.shape[-1])
    diag_indices = _jnp.arange(diag_len)
    return matrix.at[..., diag_indices, diag_indices].set(values)


# Get lower triangle and flatten to vector
def tril_to_vec(x, k=0):
    n = x.shape[-1]
    rows, cols = _jnp.tril_indices(n, k=k)
    return x[..., rows, cols]


# Get upper triangle and flatten to vector
def triu_to_vec(x, k=0):
    n = x.shape[-1]
    rows, cols = _jnp.triu_indices(n, k=k)
    return x[..., rows, cols]


def vec_to_diag(vector):
    vector = _jnp.asarray(vector)
    return vector[..., :, None] * _jnp.eye(vector.shape[-1], dtype=vector.dtype)


def mat_from_diag_triu_tril(diag, tri_upp, tri_low):
    diag = _jnp.asarray(diag)
    tri_upp = _jnp.asarray(tri_upp)
    tri_low = _jnp.asarray(tri_low)
    dtype = _jnp.result_type(diag, tri_upp, tri_low)
    diag = diag.astype(dtype)
    tri_upp = tri_upp.astype(dtype)
    tri_low = tri_low.astype(dtype)

    n = diag.shape[-1]
    i = _jnp.arange(n)
    j, k = _jnp.triu_indices(n, k=1)
    matrix = _jnp.zeros(diag.shape + (n,), dtype=dtype)
    matrix = matrix.at[..., i, i].set(diag)
    matrix = matrix.at[..., j, k].set(tri_upp)
    matrix = matrix.at[..., k, j].set(tri_low)
    return matrix
