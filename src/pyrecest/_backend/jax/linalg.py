"""JAX-based linear algebra backend."""

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
    solve,
    svd,
    qr,
    pinv,
)
from jax.scipy.linalg import (
    expm,
    sqrtm,
    polar,
    block_diag,  # For PyRecEst
)

unsupported_functions = [
    "fractional_matrix_power",
    "is_single_matrix_pd",
    "logm",
    "quadratic_assignment",
    "solve_sylvester",
]


def _raise_unsupported(*args, **kwargs):
    raise NotImplementedError("This function is not supported in this JAX backend.")


for func_name in unsupported_functions:
    globals()[func_name] = _raise_unsupported

