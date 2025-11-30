"""Jax-based computation backend.
based on implementation by Emile Mathieu
for Riemannian Score-based SDE
"""
import jax.numpy as _jnp
from jax.numpy import (
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
    # For pyrecest
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
    abs,
    arange,
    abs,
    angle,
    arange,
    arccos,
    arccosh,
    arcsin,
    arctan2,
    arctanh,
    ceil,
    copy,
    cos,
    cosh,
    divide,
    dot,
    exp,
    floor,
    imag,
    log,
    matmul,
    mod,
    ndim,
    outer,
    power,
    real,
    sign,
    sin,
    sinh,
    sqrt,
    squeeze,
    tan,
    tanh,
    trace,
    vectorize,
    empty,
    eye,
    zeros,
    linspace,
    ones,
    round,
    array_equal,
    # For Riemannian score-based SDE
    log1p,
)
from jax import vmap

def has_autodiff():
    """If allows for automatic differentiation.

    Returns
    -------
    has_autodiff : bool
    """
    return True

from ._dtype import (
    set_default_dtype, as_dtype
)

from jax import device_get as to_numpy

from jax.scipy.special import erf, gamma, polygamma, gammaln
from jax.scipy.integrate import trapezoid as trapz
from jax.scipy.integrate import trapezoid

from jax.numpy import ravel as flatten
from jax.numpy import asarray as from_numpy

from .._backend_config import jax_atol as atol
from .._backend_config import jax_rtol as rtol


from . import autodiff
from . import linalg
from . import random
from . import fft   # For PyRecEst
from . import spatial  # For PyRecEst

from jax.numpy import array


def convert_to_wider_dtype(*args, **kwargs):
    raise NotImplementedError("The function convert_to_wider_dtype is not supported in this JAX backend.")


def get_default_dtype(*args, **kwargs):
    raise NotImplementedError("The function get_default_dtype is not supported in this JAX backend.")


def get_default_cdtype(*args, **kwargs):
    raise NotImplementedError("The function get_default_cdtype is not supported in this JAX backend.")


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
    
    # Compute linear indices from multi-dimensional indices
    linear_indices = _jnp.ravel_multi_index(indices.T, target_shape)
    
    # Use JAX's indexing to place data into the output array
    out = out.at[linear_indices].set(data)
    
    return out


def is_complex(x):
    return _jnp.iscomplexobj(x)


def cast(array, dtype):
    return _jnp.asarray(array, dtype=dtype)


def ravel_tril_indices(n):
    return _jnp.tril_indices(n)


def is_array(obj):
    return isinstance(obj, _jnp.ndarray)


def get_slice(array, start, end):
    return array[start:end]

def as_dtype(array):
    """Change the data type of a given array.
    
    Parameters:
    - array: The array whose data type needs to be changed
    - dtype: The new data type
    
    Returns:
    A new array with the specified data type.
    """
    return _jnp.asarray(array, dtype=dtype)


# Check if dtype is floating-point
def is_floating(array):
    return _jnp.issubdtype(array.dtype, _jnp.floating)


# Check if dtype is boolean
def is_bool(array):
    return _jnp.issubdtype(array.dtype, _jnp.bool_)


# Matrix-vector multiplication
def matvec(matrix, vector):
    return _jnp.dot(matrix, vector)


# One-hot encoding
def one_hot(indices, depth):
    return _jnp.eye(depth)[indices]


# Scatter-add operation
def scatter_add(array, indices, updates):
    return _jnp.zeros_like(array).at[indices].add(updates)


# Set diagonal elements of a matrix
def set_diag(matrix, values):
    return matrix.at[_jnp.diag_indices_from(matrix)].set(values)


# Get lower triangle and flatten to vector
def tril_to_vec(matrix):
    return _jnp.tril(matrix).ravel()


# Get upper triangle and flatten to vector
def triu_to_vec(matrix):
    return _jnp.triu(matrix).ravel()


# Create diagonal matrix from vector
def vec_to_diag(vector):
    return _jnp.diag(vector)


# Create matrix from diagonal, upper triangular, and lower triangular parts
def mat_from_diag_triu_tril(diag, triu, tril):
    matrix = _jnp.diag(diag)
    matrix = matrix.at[_jnp.triu_indices_from(matrix, k=1)].set(triu)
    matrix = matrix.at[_jnp.tril_indices_from(matrix, k=-1)].set(tril)
    return matrix