"""JAX-based linear algebra backend."""

from operator import index as _operator_index

import jax.numpy as _jnp
import jax.scipy.linalg as _jax_scipy_linalg
import numpy as _np

from .._backend_config import jax_atol as atol

_AXIS_TYPE_ERROR = "axis must be None, an integer, or a tuple of integers"


def _as_linalg_array(value):
    """Convert PyRecEst array-like inputs before calling raw JAX linalg."""
    return _jnp.asarray(value)


def _as_integer_scalar(value, name):
    """Return a Python integer for JAX static integer arguments."""
    if isinstance(value, (bool, _np.bool_)):
        raise TypeError(f"{name} must be an integer scalar")
    try:
        return _operator_index(value)
    except TypeError:
        pass

    value_array = _jnp.asarray(value)
    if value_array.shape != () or value_array.dtype == _jnp.bool_:
        raise TypeError(f"{name} must be an integer scalar")
    if not _jnp.issubdtype(value_array.dtype, _jnp.integer):
        raise TypeError(f"{name} must be an integer scalar")
    return int(value_array.item())


def _norm_axis_entry(axis) -> int:
    """Return one non-boolean NumPy-style linalg.norm axis."""
    if isinstance(axis, (bool, _np.bool_)):
        raise TypeError(_AXIS_TYPE_ERROR)
    try:
        return _operator_index(axis)
    except TypeError:
        pass

    axis_array = _jnp.asarray(axis)
    if axis_array.shape != () or axis_array.dtype == _jnp.bool_:
        raise TypeError(_AXIS_TYPE_ERROR)
    if not _jnp.issubdtype(axis_array.dtype, _jnp.integer):
        raise TypeError(_AXIS_TYPE_ERROR)
    return int(axis_array.item())


def _normalize_norm_axis(axis):
    """Normalize JAX linalg.norm axes before they become static arguments."""
    if axis is None:
        return None
    if isinstance(axis, (list, tuple)):
        return tuple(_norm_axis_entry(one_axis) for one_axis in axis)

    axis_array = _jnp.asarray(axis)
    if axis_array.shape == ():
        return _norm_axis_entry(axis)
    if axis_array.ndim == 1:
        return tuple(_norm_axis_entry(one_axis) for one_axis in axis_array.tolist())
    raise TypeError(_AXIS_TYPE_ERROR)


def cholesky(a, *args, **kwargs):
    return _jnp.linalg.cholesky(_as_linalg_array(a), *args, **kwargs)


def det(a, *args, **kwargs):
    return _jnp.linalg.det(_as_linalg_array(a), *args, **kwargs)


def eig(a, *args, **kwargs):
    return _jnp.linalg.eig(_as_linalg_array(a), *args, **kwargs)


def eigh(a, *args, **kwargs):
    return _jnp.linalg.eigh(_as_linalg_array(a), *args, **kwargs)


def eigvalsh(a, *args, **kwargs):
    return _jnp.linalg.eigvalsh(_as_linalg_array(a), *args, **kwargs)


def inv(a, *args, **kwargs):
    return _jnp.linalg.inv(_as_linalg_array(a), *args, **kwargs)


def matrix_power(a, n):
    return _jnp.linalg.matrix_power(_as_linalg_array(a), _as_integer_scalar(n, "n"))


def matrix_rank(a, *args, **kwargs):
    return _jnp.linalg.matrix_rank(_as_linalg_array(a), *args, **kwargs)


def norm(x, ord=None, axis=None, keepdims=False):
    return _jnp.linalg.norm(
        _as_linalg_array(x),
        ord=ord,
        axis=_normalize_norm_axis(axis),
        keepdims=keepdims,
    )


def pinv(a, *args, **kwargs):
    return _jnp.linalg.pinv(_as_linalg_array(a), *args, **kwargs)


def qr(a, *args, **kwargs):
    """Compute QR decomposition with NumPy-compatible legacy mode handling."""
    a = _as_linalg_array(a)
    if args:
        mode = args[0]
        if len(args) != 1 or "mode" in kwargs or mode != "economic":
            return _jnp.linalg.qr(a, *args, **kwargs)
        raw_qr = _jnp.linalg.qr(a, "raw", **kwargs)
    else:
        mode = kwargs.get("mode", "reduced")
        if mode != "economic":
            return _jnp.linalg.qr(a, **kwargs)
        kwargs = dict(kwargs)
        kwargs["mode"] = "raw"
        raw_qr = _jnp.linalg.qr(a, **kwargs)

    # NumPy's deprecated ``economic`` mode returns the transposed raw Householder
    # reflector matrix. JAX exposes that matrix as ``Q`` for ``mode="raw"``.
    return _jnp.swapaxes(raw_qr.Q, -1, -2)


def solve(a, b, *args, **kwargs):
    return _jnp.linalg.solve(_as_linalg_array(a), _as_linalg_array(b), *args, **kwargs)


def svd(a, *args, **kwargs):
    return _jnp.linalg.svd(_as_linalg_array(a), *args, **kwargs)


def block_diag(*arrs):
    return _jax_scipy_linalg.block_diag(*(_as_linalg_array(arr) for arr in arrs))


def expm(a, *args, **kwargs):
    return _jax_scipy_linalg.expm(_as_linalg_array(a), *args, **kwargs)


def polar(a, *args, **kwargs):
    return _jax_scipy_linalg.polar(_as_linalg_array(a), *args, **kwargs)


def sqrtm(a, *args, **kwargs):
    return _jax_scipy_linalg.sqrtm(_as_linalg_array(a), *args, **kwargs)


def solve_sylvester(a, b, q, *args, **kwargs):
    """Solve ``A X + X B = Q`` using JAX arrays and JAX SciPy."""
    return _jax_scipy_linalg.solve_sylvester(
        _as_linalg_array(a),
        _as_linalg_array(b),
        _as_linalg_array(q),
        *args,
        **kwargs,
    )


def _unsupported_function(name):
    def _raise_unsupported(*args, **kwargs):
        _ = args, kwargs
        raise NotImplementedError(f"{name} is not available in this JAX backend.")

    _raise_unsupported.__name__ = name
    _raise_unsupported.__qualname__ = name
    _raise_unsupported.__doc__ = f"JAX-backend placeholder for ``{name}``."
    return _raise_unsupported


fractional_matrix_power = _unsupported_function("fractional_matrix_power")
logm = _unsupported_function("logm")
quadratic_assignment = _unsupported_function("quadratic_assignment")


def is_single_matrix_pd(mat):
    """Check if a 2D square matrix is positive definite."""
    mat = _as_linalg_array(mat)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return False

    if mat.dtype in (_jnp.complex64, _jnp.complex128):
        is_hermitian = _jnp.all(_jnp.abs(mat - _jnp.conj(_jnp.transpose(mat))) < atol)
        eigvals = _jnp.linalg.eigvalsh(mat)
        return _jnp.logical_and(is_hermitian, _jnp.min(_jnp.real(eigvals)) > 0)

    is_symmetric = _jnp.all(_jnp.abs(mat - _jnp.transpose(mat)) < atol)
    factor = _jnp.linalg.cholesky(mat)
    return _jnp.logical_and(is_symmetric, _jnp.all(_jnp.isfinite(factor)))
