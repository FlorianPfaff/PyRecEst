from ._dispatch import _common
from ._dispatch import numpy as _np
from ._dispatch import scipy as _scipy

_to_ndarray = _common.to_ndarray
atol = _common.atol
rtol = _common.rtol


def _transpose(array):
    axes = list(range(0, array.ndim))
    axes[-2], axes[-1] = axes[-1], axes[-2]
    return _np.transpose(array, axes=axes)


def _adjoint(array):
    return _np.conj(_transpose(array))


def _as_scipy_linalg_array(x):
    """Return an array with a dtype suitable for SciPy linalg routines."""

    array = _np.asarray(x)
    if array.dtype.kind in ("f", "c"):
        return array
    return array.astype(_np.float64)


def _complex_dtype_like(input_array):
    if input_array.dtype in (_np.float32, _np.complex64):
        return _np.complex64
    if input_array.dtype in (_np.float64, _np.complex128):
        return _np.complex128
    return None


def _cast_scipy_linalg_result_to_input_dtype(result, input_array):
    if result.dtype.kind == "f":
        if input_array.dtype.kind == "f" and result.dtype != input_array.dtype:
            return _common.cast(result, input_array.dtype)
    elif result.dtype.kind == "c":
        if input_array.dtype.kind == "c":
            target_dtype = input_array.dtype
        elif input_array.dtype.kind == "f":
            target_dtype = _complex_dtype_like(input_array)
        else:
            target_dtype = None
        if target_dtype is not None and result.dtype != target_dtype:
            return _common.cast(result, target_dtype)
    return result


def _empty_batched_square_matrix_result(array):
    """Return an empty matrix-function result for a valid empty batch."""

    if array.ndim > 2 and 0 in array.shape[:-2] and array.shape[-2] == array.shape[-1]:
        return _np.empty_like(array)
    return None


def _is_symmetric(x, tol=atol):
    return (_np.abs(x - _transpose(x)) < tol).all()


def _is_hermitian(x, tol=atol):
    return (_np.abs(x - _adjoint(x)) < tol).all()


def _sylvester_candidate_is_accurate(a, b, q, candidate):
    residual_target = a @ candidate + candidate @ b
    return _np.all(_np.isclose(residual_target, q, atol=atol, rtol=rtol))


def _broadcast_shapes(*shapes):
    """Return the NumPy-style broadcast shape without allocating arrays."""

    reversed_shapes = [tuple(reversed(shape)) for shape in shapes]
    result = []
    for index in range(max((len(shape) for shape in reversed_shapes), default=0)):
        dimensions = {
            shape[index] if index < len(shape) else 1 for shape in reversed_shapes
        }
        dimensions.discard(1)
        if len(dimensions) > 1:
            raise ValueError("operands could not be broadcast together")
        result.append(dimensions.pop() if dimensions else 1)
    return tuple(reversed(result))


def _empty_sylvester_batch_result(a, b, q):
    """Return a broadcast, SciPy-compatible empty Sylvester result."""

    if a.ndim < 2 or b.ndim < 2 or q.ndim < 2:
        return None
    if a.shape[-2] != a.shape[-1] or b.shape[-2] != b.shape[-1]:
        return None
    if q.shape[-2:] != (a.shape[-1], b.shape[-1]):
        return None

    try:
        batch_shape = _broadcast_shapes(
            a.shape[:-2],
            b.shape[:-2],
            q.shape[:-2],
        )
    except ValueError:
        return None
    if 0 not in batch_shape:
        return None

    promoted_dtypes = []
    for array in (a, b, q):
        dtype = array.dtype
        if dtype.kind in "iu":
            dtype = _np.dtype(_np.float64)
        elif dtype.kind == "b" or dtype == _np.dtype(_np.float16):
            dtype = _np.dtype(_np.float32)
        elif dtype.kind not in "fc":
            return None
        promoted_dtypes.append(dtype)

    return _np.empty(
        batch_shape + q.shape[-2:],
        dtype=_np.result_type(*promoted_dtypes),
    )


_diag_vec = _np.vectorize(_np.diag, signature="(n)->(n,n)")
_logm_vec = _np.vectorize(_scipy.linalg.logm, signature="(n,m)->(n,m)")


def logm(x):
    x = _as_scipy_linalg_array(x)
    empty_result = _empty_batched_square_matrix_result(x)
    if empty_result is not None:
        return empty_result
    if _is_symmetric(x) and x.dtype not in [_np.complex64, _np.complex128]:
        eigvals, eigvecs = _np.linalg.eigh(x)
        if (eigvals > 0).all():
            eigvals = _np.log(eigvals)
            eigvals = _diag_vec(eigvals)
            transp_eigvecs = _transpose(eigvecs)
            result = _np.matmul(eigvecs, eigvals)
            result = _np.matmul(result, transp_eigvecs)
        else:
            result = _logm_vec(x)
    else:
        result = _logm_vec(x)

    return _cast_scipy_linalg_result_to_input_dtype(result, x)


def solve_sylvester(a, b, q, tol=atol):
    a = _np.asarray(a)
    b = _np.asarray(b)
    q = _np.asarray(q)

    empty_result = _empty_sylvester_batch_result(a, b, q)
    if empty_result is not None:
        return empty_result

    if a.shape == b.shape:
        if _np.all(_np.isclose(a, b)) and _is_hermitian(a, tol=tol):
            eigvals, eigvecs = _np.linalg.eigh(a)
            if _np.all(eigvals >= tol):
                adjoint_eigvecs = _adjoint(eigvecs)
                tilde_q = adjoint_eigvecs @ q @ eigvecs
                tilde_x = tilde_q / (eigvals[..., :, None] + eigvals[..., None, :])
                candidate = eigvecs @ tilde_x @ adjoint_eigvecs
                if _sylvester_candidate_is_accurate(a, b, q, candidate):
                    return candidate

    return _np.vectorize(
        _scipy.linalg.solve_sylvester, signature="(m,m),(n,n),(m,n)->(m,n)"
    )(a, b, q)


def sqrtm(x):
    x = _as_scipy_linalg_array(x)
    empty_result = _empty_batched_square_matrix_result(x)
    if empty_result is not None:
        return empty_result
    result = _np.vectorize(_scipy.linalg.sqrtm, signature="(n,m)->(n,m)")(x)
    return _cast_scipy_linalg_result_to_input_dtype(result, x)


def quadratic_assignment(a, b, options=None):
    return list(_scipy.optimize.quadratic_assignment(a, b, options=options).col_ind)


def qr(a, mode="reduced"):
    if 0 in _np.shape(a)[:-2]:
        return _np.linalg.qr(a, mode=mode)
    if mode == "r":
        signature = "(n,m)->(k,m)"
    elif mode == "raw":
        signature = "(n,m)->(m,n),(k)"
    elif mode == "economic":
        signature = "(n,m)->(n,m)"
    else:
        signature = "(n,m)->(n,k),(k,m)"
    return _np.vectorize(_np.linalg.qr, signature=signature, excluded=["mode"])(
        a, mode=mode
    )


def is_single_matrix_pd(mat):
    """Check if a finite 2D square matrix is positive definite."""
    mat = _np.asarray(mat)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return False
    if not _np.all(_np.isfinite(mat)):
        return False
    if mat.dtype in [_np.complex64, _np.complex128]:
        if not _is_hermitian(mat):
            return False
        eigvals = _np.linalg.eigvalsh(mat)
        return _np.all(_np.isfinite(eigvals)) and _np.min(_np.real(eigvals)) > 0
    if not _is_symmetric(mat):
        return False
    try:
        factor = _np.linalg.cholesky(mat)
        return bool(_np.all(_np.isfinite(factor)))
    except _np.linalg.LinAlgError as e:
        if e.args[0] == "Matrix is not positive definite":
            return False
        raise e


def fractional_matrix_power(A, t):
    A = _as_scipy_linalg_array(A)
    empty_result = _empty_batched_square_matrix_result(A)
    if empty_result is not None:
        return empty_result
    result = _np.vectorize(
        lambda one_matrix: _scipy.linalg.fractional_matrix_power(one_matrix, t),
        signature="(n,n)->(n,n)",
    )(A)
    return _cast_scipy_linalg_result_to_input_dtype(result, A)


def polar(a, side="right"):
    """Polar decomposition of a square or rectangular matrix."""
    if side not in {"left", "right"}:
        raise ValueError("`side` must be either 'right' or 'left'")

    a = _as_scipy_linalg_array(a)
    if 0 in a.shape[:-2]:
        positive_dim = a.shape[-2] if side == "left" else a.shape[-1]
        positive_shape = a.shape[:-2] + (positive_dim, positive_dim)
        return _np.empty_like(a), _np.empty(positive_shape, dtype=a.dtype)

    signature = "(m,n)->(m,n),(m,m)" if side == "left" else "(m,n)->(m,n),(n,n)"
    return _np.vectorize(_scipy.linalg.polar, signature=signature, excluded=["side"])(
        a, side=side
    )


def solve(a, b):
    """
    Solve a linear matrix equation, or system of linear scalar equations.

    Computes the "exact" solution, `x`, of the well-determined, i.e., full
    rank, linear matrix equation `ax = b`.
    Parameters
    ----------
    a : array-like, shape=[..., M, M]
        Coefficient matrix.
    b : array-like, shape=[..., M] or [..., M, K]
        Ordinate or "dependent variable" values.

    Returns
    -------
    x : array-like, shape=[..., M] or [..., M, K]
        Solution to the system a x = b.
    """
    a = _np.asarray(a)
    b = _np.asarray(b)

    vector_rhs = b.ndim == a.ndim - 1
    if vector_rhs:
        b = _np.expand_dims(b, axis=-1)

    res = _np.linalg.solve(a, b)
    if vector_rhs:
        return res[..., 0]

    return res
