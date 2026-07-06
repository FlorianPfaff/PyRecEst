import jax.numpy as _jnp
from pyrecest._backend._dtype_utils import _pre_set_default_dtype

# Mapping of string dtype representations to JAX dtypes
MAP_DTYPE = {
    "bool": _jnp.bool_,
    "bool_": _jnp.bool_,
    "uint8": _jnp.uint8,
    "int8": _jnp.int8,
    "int16": _jnp.int16,
    "int32": _jnp.int32,
    "int64": _jnp.int64,
    "bfloat16": _jnp.bfloat16,
    "float16": _jnp.float16,
    "float32": _jnp.float32,
    "float64": _jnp.float64,
    "complex64": _jnp.complex64,
    "complex128": _jnp.complex128,
}


def _dtype_key(value):
    try:
        return str(_jnp.dtype(value))
    except (TypeError, ValueError):
        return str(value).rsplit(".", maxsplit=1)[-1].removesuffix("'>")


def as_dtype(value):
    """
    Transform string or dtype-like value into JAX dtype.

    Parameters
    ----------
    value : str or dtype-like
        String or object representing the dtype to be converted.

    Returns
    -------
    dtype : jnp.dtype
        JAX dtype object corresponding to the input value.
    """
    return MAP_DTYPE[_dtype_key(value)]


set_default_dtype = _pre_set_default_dtype(as_dtype)
