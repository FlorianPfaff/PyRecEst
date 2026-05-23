"""JAX-based linear algebra backend."""

import jax.numpy as _jnp
from jax.numpy.linalg import (  # NOQA
    cholesky,
    det,
    eig,
    eigh,
    eigvalsh,
    inv,
    matrix_power,
    matrix_rank,
    norm,
    pinv,
    qr,
    solve,
    svd,
)
from jax.scipy.linalg import block_diag  # For PyRecEst
from jax.scipy.linalg import (
    expm,
    polar,
    sqrtm,
)

from .._backend_config import jax_atol as atol

unsupported_functions = [
    "fractional_matrix_power",
    "logm",
    "quadratic_assignment",
    "solve_sylvester",
]


def _raise_unsupported(*args, **kwargs):
    raise NotImplementedError("This function is not supported in this JAX backend.")


def is_single_matrix_pd(mat):
    """Check if a 2D square matrix is positive definite."""
    mat = _jnp.asarray(mat)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return False

    if mat.dtype in (_jnp.complex64, _jnp.complex128):
        is_hermitian = _jnp.all(
            _jnp.abs(mat - _jnp.conj(_jnp.transpose(mat))) < atol
        )
        eigvals = _jnp.linalg.eigvalsh(mat)
        return _jnp.logical_and(is_hermitian, _jnp.min(_jnp.real(eigvals)) > 0)

    factor = _jnp.linalg.cholesky(mat)
    return _jnp.all(_jnp.isfinite(factor))


for func_name in unsupported_functions:
    globals()[func_name] = _raise_unsupported
