"""Numpy based linear algebra backend."""

import numpy as _np
import scipy as _scipy
from numpy.linalg import (
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
    svd,
)
from scipy.linalg import block_diag  # For PyRecEst
from scipy.linalg import (
    expm,
)

from .._shared_numpy.linalg import fractional_matrix_power as _fractional_matrix_power
from .._shared_numpy.linalg import (
    is_single_matrix_pd,
)
from .._shared_numpy.linalg import logm as _logm
from .._shared_numpy.linalg import (
    polar,
    qr,
    quadratic_assignment,
    solve,
    solve_sylvester,
)
from .._shared_numpy.linalg import sqrtm as _sqrtm


def _empty_zero_by_zero_matrix_result(value):
    """Return a SciPy-compatible result for trailing zero-by-zero matrices."""
    matrix = _np.asarray(value)
    if matrix.ndim < 2 or matrix.shape[-2:] != (0, 0):
        return None
    if matrix.dtype.kind not in ("f", "c"):
        matrix = matrix.astype(_np.float64)
    return _np.empty_like(matrix)


def logm(x):
    """Compute a matrix logarithm, including zero-by-zero matrices."""
    empty_result = _empty_zero_by_zero_matrix_result(x)
    if empty_result is not None:
        return empty_result
    return _logm(x)


def sqrtm(x):
    """Compute a matrix square root, including zero-by-zero matrices."""
    empty_result = _empty_zero_by_zero_matrix_result(x)
    if empty_result is not None:
        return empty_result
    return _sqrtm(x)


def fractional_matrix_power(a, t):
    """Compute a fractional matrix power, including zero-by-zero matrices."""
    empty_result = _empty_zero_by_zero_matrix_result(a)
    if empty_result is not None:
        return empty_result
    return _fractional_matrix_power(a, t)
