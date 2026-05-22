"""Jax-based computation backend.
based on implementation by Emile Mathieu
for Riemannian Score-based SDE
"""

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
    meshgrid,
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
    trace,
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

    # Check if we need to add a dimension
    if x.ndim == to_ndim - 1:
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
    # Ensure indices and values are iterable
    if isinstance(indices, (int, tuple)):
        indices = [indices]
    if not isinstance(values, list):
        values = [values] * len(indices)

    # Check if we need to raise errors for mismatch in values and indices lengths
    if len(values) != 1 and len(values) != len(indices):
        raise ValueError("Either one value or as many values as indices required")

    # Handling assignment with index update
    x_new = x.at[indices].set(values)

    return x_new


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
    # Ensure indices and values are iterable
    if isinstance(indices, (int, tuple)):
        indices = [indices]
    if not isinstance(values, list):
        values = [values] * len(indices)

    # Check if the number of values matches the number of indices, or there's exactly one value
    if len(values) != 1 and len(values) != len(indices):
        raise ValueError("Either one value or as many values as indices required")

    # Handling addition with index update
    for idx, val in zip(indices, values):
        x = x.at[idx].add(val)

    return x


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
    if ignore_div_zero is False:
        return _jnp.divide(a, b)

    a_arr, b_arr = _jnp.asarray(a), _jnp.asarray(b)
    quotient = _jnp.divide(a_arr, b_arr)
    return _jnp.where(b_arr != 0, quotient, _jnp.zeros_like(quotient))


def dot(a, b):
    a = _jnp.asarray(a)
    b = _jnp.asarray(b)

    if b.ndim == 1:
        return _jnp.dot(a, b)
    if a.ndim == 1:
        return _jnp.dot(a, b.T)
    return _jnp.einsum("...i,...i->...", a, b)


def outer(a, b):
    a = _jnp.asarray(a)
    b = _jnp.asarray(b)

    if a.ndim > 1 and b.ndim > 1:
        return _jnp.einsum("...i,...j->...ij", a, b)

    result = _jnp.multiply.outer(a, b)
    if b.ndim > 1:
        result = _jnp.swapaxes(result, 0, -2)
    return result


# Matrix-vector multiplication
def matvec(matrix, vector):
    matrix = _jnp.asarray(matrix)
    vector = _jnp.asarray(vector)

    if vector.ndim == 1:
        return _jnp.matmul(matrix, vector)
    if matrix.ndim == 2:
        return _jnp.matmul(matrix, vector.T).T
    return _jnp.einsum("...ij,...j->...i", matrix, vector)


# One-hot encoding
def one_hot(indices, depth):
    return _jnp.eye(depth)[indices]


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
    diag_len = matrix.shape[-2] if matrix.shape[-2] < matrix.shape[-1] else matrix.shape[-1]
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
