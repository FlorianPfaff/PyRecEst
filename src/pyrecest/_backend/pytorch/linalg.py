"""Pytorch based linear algebra backend."""

from operator import index as _operator_index

import numpy as _np
import scipy as _scipy
import torch as _torch

from .._backend_config import pytorch_atol as atol
from ..numpy import linalg as _gsnplinalg
from ._common import array, cast
from ._dtype import (
    get_default_dtype,
    is_complex,
    is_floating,
)

# The public backend facade exposes NumPy-style helpers. Keep array-like
# coercion local to this module instead of re-exporting raw torch.linalg
# functions, because raw torch.linalg rejects Python lists and integer arrays.


def _as_numpy_no_grad(value):
    """Return a CPU NumPy view/copy for SciPy bridge functions."""
    if isinstance(value, _torch.Tensor):
        return value.detach().resolve_conj().resolve_neg().cpu().numpy()
    return _np.asarray(value)


def _torch_as_like(value, like):
    """Convert a NumPy/SciPy result back to the input tensor's device and dtype."""
    if isinstance(like, _torch.Tensor):
        result = _torch.as_tensor(value, device=like.device)
        if result.dtype.is_floating_point and like.dtype.is_floating_point:
            return result.to(dtype=like.dtype)
        if result.dtype.is_complex and like.dtype.is_complex:
            return result.to(dtype=like.dtype)
        return result
    return _torch.from_numpy(_np.asarray(value))


_COMPLEX_DTYPE_FOR_TENSOR_DTYPE = {
    _torch.float32: _np.complex64,
    _torch.float64: _np.complex128,
    _torch.complex64: _np.complex64,
    _torch.complex128: _np.complex128,
}

_AXIS_TYPE_ERROR = "axis must be None, an integer, or a tuple of integers"


def _default_linalg_dtype():
    dtype = get_default_dtype()
    if dtype in (_torch.float32, _torch.float64):
        return dtype
    if dtype == _np.dtype("float32"):
        return _torch.float32
    if dtype == _np.dtype("float64"):
        return _torch.float64
    return _torch.float64


def _as_linalg_tensor(value):
    """Convert array-like values to a floating/complex tensor for torch.linalg."""
    tensor = array(value)
    if not is_floating(tensor) and not is_complex(tensor):
        tensor = cast(tensor, dtype=_default_linalg_dtype())
    return tensor


def _has_empty_matrix_batch(matrix):
    """Return whether a matrix batch has a zero-length leading dimension."""
    return matrix.ndim > 2 and 0 in matrix.shape[:-2]


def _empty_square_matrix_batch_result(matrix, function_name):
    """Return a shape-preserving empty result for a square matrix batch."""
    if not _has_empty_matrix_batch(matrix):
        return None
    if matrix.shape[-2] != matrix.shape[-1]:
        raise ValueError(f"{function_name} requires square matrices")
    return matrix.new_empty(matrix.shape)


def _is_boolean_scalar(value):
    """Return whether ``value`` is a scalar boolean exponent candidate."""
    if isinstance(value, (bool, _np.bool_)):
        return True
    if isinstance(value, _np.ndarray):
        return value.shape == () and value.dtype == _np.bool_
    if _torch.is_tensor(value):
        return value.ndim == 0 and value.dtype == _torch.bool
    return False


def _as_integer_scalar(value, name):
    """Return a non-boolean Python integer for Torch integer-scalar arguments."""
    if _is_boolean_scalar(value):
        raise TypeError(f"{name} must be an integer scalar, not boolean")
    try:
        return _operator_index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer scalar") from exc


def _as_norm_axis_entry(axis) -> int:
    """Return one non-boolean PyTorch linalg.norm axis entry."""
    if _is_boolean_scalar(axis):
        raise TypeError(_AXIS_TYPE_ERROR)
    try:
        return _operator_index(axis)
    except TypeError as exc:
        raise TypeError(_AXIS_TYPE_ERROR) from exc


def _common_linalg_dtype(*tensors):
    """Return a common floating/complex dtype for torch.linalg operations."""
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = _torch.promote_types(dtype, tensor.dtype)
    if dtype.is_floating_point or dtype.is_complex:
        return dtype
    return _default_linalg_dtype()


def _preferred_linalg_device(*values):
    """Return the non-CPU tensor device to preserve, falling back to any tensor."""
    non_cpu_device = next(
        (
            value.device
            for value in values
            if _torch.is_tensor(value) and value.device.type != "cpu"
        ),
        None,
    )
    if non_cpu_device is not None:
        return non_cpu_device
    return next((value.device for value in values if _torch.is_tensor(value)), None)


def _out_kwargs(out):
    return {} if out is None else {"out": out}


def _linalg_tolerance_dtype(reference):
    """Return the real tolerance dtype matching a linalg input tensor."""
    if not isinstance(reference, _torch.Tensor):
        return None
    if reference.dtype == _torch.complex64:
        return _torch.float32
    if reference.dtype == _torch.complex128:
        return _torch.float64
    if reference.dtype.is_floating_point:
        return reference.dtype
    return None


def _normalize_linalg_tolerance(value, reference=None):
    """Return PyTorch-compatible tolerances from NumPy scalar/array inputs."""
    if value is None or _torch.is_tensor(value):
        return value
    if isinstance(value, _np.generic):
        return value.item()
    if isinstance(value, _np.ndarray):
        if value.ndim == 0:
            return value.item()
        kwargs = {}
        if isinstance(reference, _torch.Tensor):
            kwargs["device"] = reference.device
            dtype = _linalg_tolerance_dtype(reference)
            if dtype is not None:
                kwargs["dtype"] = dtype
        return _torch.as_tensor(value, **kwargs)
    if isinstance(value, (list, tuple)):
        kwargs = {}
        if isinstance(reference, _torch.Tensor):
            kwargs["device"] = reference.device
            dtype = _linalg_tolerance_dtype(reference)
            if dtype is not None:
                kwargs["dtype"] = dtype
        return _torch.as_tensor(value, **kwargs)
    return value


def cholesky(a, upper=False, out=None):
    """Compute a Cholesky factor after PyRecEst-style array-like promotion."""
    return _torch.linalg.cholesky(_as_linalg_tensor(a), upper=upper, **_out_kwargs(out))


def det(a, out=None):
    """Compute a determinant after PyRecEst-style array-like promotion."""
    return _torch.linalg.det(_as_linalg_tensor(a), **_out_kwargs(out))


def eig(a, out=None):
    """Compute eigenvalues/eigenvectors after array-like input promotion."""
    return _torch.linalg.eig(_as_linalg_tensor(a), **_out_kwargs(out))


def eigh(a, UPLO="L", out=None):
    """Compute Hermitian eigenpairs after array-like input promotion."""
    return _torch.linalg.eigh(_as_linalg_tensor(a), UPLO=UPLO, **_out_kwargs(out))


def eigvalsh(a, UPLO="L", out=None):
    """Compute Hermitian eigenvalues after array-like input promotion."""
    return _torch.linalg.eigvalsh(_as_linalg_tensor(a), UPLO=UPLO, **_out_kwargs(out))


def inv(a, out=None):
    """Invert a matrix after PyRecEst-style array-like input promotion."""
    return _torch.linalg.inv(_as_linalg_tensor(a), **_out_kwargs(out))


def expm(a):
    """Compute the matrix exponential after array-like input promotion."""
    return _torch.linalg.matrix_exp(_as_linalg_tensor(a))


def matrix_power(a, n):
    """Raise a matrix to an integer power after array-like input promotion."""
    return _torch.linalg.matrix_power(_as_linalg_tensor(a), _as_integer_scalar(n, "n"))


def pinv(a, rcond=None, hermitian=False, *, atol=None, rtol=None, out=None):
    """Compute the Moore-Penrose pseudoinverse after array-like input promotion."""
    if rcond is not None:
        if rtol is not None:
            raise TypeError("pinv() got both 'rcond' and 'rtol'")
        rtol = rcond
    a = _as_linalg_tensor(a)
    atol = _normalize_linalg_tolerance(atol, a)
    rtol = _normalize_linalg_tolerance(rtol, a)
    return _torch.linalg.pinv(
        a,
        atol=atol,
        rtol=rtol,
        hermitian=hermitian,
        **_out_kwargs(out),
    )


def block_diag(*arrs):
    """Build a block diagonal tensor from PyRecEst-style array-like inputs."""
    return _torch.block_diag(*(array(arr) for arr in arrs))


class _Logm(_torch.autograd.Function):
    """Torch autograd function for matrix logarithm."""

    @staticmethod
    def _logm(x):
        mat_log = _gsnplinalg.logm(_as_numpy_no_grad(x))
        if mat_log.dtype.kind == "c":
            target_complex_dtype = _COMPLEX_DTYPE_FOR_TENSOR_DTYPE.get(x.dtype)
            if target_complex_dtype is not None:
                mat_log = mat_log.astype(target_complex_dtype, copy=False)
        return _torch_as_like(mat_log, x)

    @staticmethod
    def forward(ctx, tensor):
        """Apply matrix logarithm to a tensor."""
        ctx.save_for_backward(tensor)
        return _Logm._logm(tensor)

    @staticmethod
    def backward(ctx, grad):
        """Run gradients backward."""
        (tensor,) = ctx.saved_tensors

        tensor_H = tensor.transpose(-2, -1).conj().to(grad.dtype)
        n = tensor.size(-1)
        bshape = tensor.shape[:-2] + (2 * n, 2 * n)
        backward_tensor = _torch.zeros(*bshape, dtype=grad.dtype, device=grad.device)
        backward_tensor[..., :n, :n] = tensor_H
        backward_tensor[..., n:, n:] = tensor_H
        backward_tensor[..., :n, n:] = grad

        return _Logm._logm(backward_tensor).to(tensor.dtype)[..., :n, n:]


def logm(x):
    """Compute the matrix logarithm after array-like input promotion."""
    return _Logm.apply(_as_linalg_tensor(x))


def sqrtm(x):
    x = _as_linalg_tensor(x)
    empty_result = _empty_square_matrix_batch_result(x, "sqrtm")
    if empty_result is not None:
        return empty_result

    x_np = _as_numpy_no_grad(x)
    np_sqrtm = _np.vectorize(_scipy.linalg.sqrtm, signature="(n,m)->(n,m)")(x_np)
    if np_sqrtm.dtype.kind == "c":
        target_complex_dtype = _COMPLEX_DTYPE_FOR_TENSOR_DTYPE.get(x.dtype)
        if target_complex_dtype is not None:
            np_sqrtm = np_sqrtm.astype(target_complex_dtype, copy=False)

    return _torch_as_like(np_sqrtm, x)


def svd(x, full_matrices=True, compute_uv=True):
    x = _as_linalg_tensor(x)
    if compute_uv:
        return _torch.linalg.svd(x, full_matrices=full_matrices)

    return _torch.linalg.svdvals(x)


def _normalize_norm_axis(axis):
    """Return a PyTorch-compatible norm dimension from NumPy-style axis input."""
    if axis is None:
        return None
    if _is_boolean_scalar(axis):
        raise TypeError(_AXIS_TYPE_ERROR)
    if _torch.is_tensor(axis):
        if axis.dtype == _torch.bool:
            raise TypeError(_AXIS_TYPE_ERROR)
        if axis.ndim == 0:
            return _as_norm_axis_entry(axis.item())
        if axis.ndim != 1:
            raise TypeError(_AXIS_TYPE_ERROR)
        axis = axis.detach().cpu().tolist()
    elif isinstance(axis, _np.ndarray):
        if axis.dtype == _np.bool_:
            raise TypeError(_AXIS_TYPE_ERROR)
        if axis.ndim == 0:
            return _as_norm_axis_entry(axis.item())
        if axis.ndim != 1:
            raise TypeError(_AXIS_TYPE_ERROR)
        axis = axis.tolist()
    elif isinstance(axis, (int, _np.integer)):
        return _as_norm_axis_entry(axis)

    if isinstance(axis, (list, tuple)):
        return tuple(_as_norm_axis_entry(one_axis) for one_axis in axis)
    return _as_norm_axis_entry(axis)


def norm(x, ord=None, axis=None, keepdims=False):
    x = _as_linalg_tensor(x)
    axis = _normalize_norm_axis(axis)
    return _torch.linalg.norm(x, ord=ord, dim=axis, keepdim=keepdims)


def matrix_rank(a, tol=None, hermitian=False, *, rtol=None, atol=None, **kwargs):
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(
            f"matrix_rank() got unexpected keyword argument(s): {unexpected}"
        )
    if tol is not None:
        if atol is not None:
            raise TypeError("matrix_rank() got both 'tol' and 'atol'")
        atol = tol

    a = _as_linalg_tensor(a)
    atol = _normalize_linalg_tolerance(atol, a)
    rtol = _normalize_linalg_tolerance(rtol, a)
    return _torch.linalg.matrix_rank(a, atol=atol, rtol=rtol, hermitian=hermitian)


def solve(a, b):
    """Solve a linear system with PyRecEst-compatible array-like inputs."""
    device = _preferred_linalg_device(a, b)
    a = _as_linalg_tensor(a)
    b = _as_linalg_tensor(b)
    if device is not None:
        a = a.to(device=device)
        b = b.to(device=device)
    common_dtype = _common_linalg_dtype(a, b)
    a = a.to(dtype=common_dtype)
    b = b.to(dtype=common_dtype)
    return _torch.linalg.solve(a, b)


def quadratic_assignment(a, b, options=None):
    return list(
        _scipy.optimize.quadratic_assignment(
            _as_numpy_no_grad(a), _as_numpy_no_grad(b), options=options
        ).col_ind
    )


def qr(a, mode="reduced"):
    """Compute QR decomposition with NumPy-compatible mode handling."""
    a = _as_linalg_tensor(a)
    if mode == "full":
        # NumPy still accepts the deprecated ``full`` alias for ``reduced``.
        mode = "reduced"
    if mode in {"reduced", "complete"}:
        return _torch.linalg.qr(a, mode=mode)
    if mode == "r":
        return _torch.linalg.qr(a, mode=mode).R
    if mode in {"raw", "economic"}:
        result = _np.linalg.qr(_as_numpy_no_grad(a), mode=mode)
        if mode == "raw":
            h, tau = result
            return _torch_as_like(h, a), _torch_as_like(tau, a)
        return _torch_as_like(result, a)
    raise ValueError(f"Unrecognized mode {mode!r}")


def _sylvester_candidate_is_accurate(a, b, q, candidate):
    """Return whether a shortcut candidate solves the original equation."""
    return _torch.allclose(
        a @ candidate + candidate @ b,
        q,
        atol=1e-6,
        rtol=1e-6,
    )


def _empty_sylvester_batch_result(a, b, q):
    """Return a broadcast, shape-preserving empty Sylvester result."""

    if a.ndim < 2 or b.ndim < 2 or q.ndim < 2:
        return None
    if a.shape[-2] != a.shape[-1] or b.shape[-2] != b.shape[-1]:
        return None
    if q.shape[-2:] != (a.shape[-1], b.shape[-1]):
        return None

    try:
        batch_shape = _torch.broadcast_shapes(
            a.shape[:-2],
            b.shape[:-2],
            q.shape[:-2],
        )
    except (RuntimeError, ValueError):
        return None
    if 0 not in batch_shape:
        return None

    return q.new_empty(tuple(batch_shape) + tuple(q.shape[-2:]))


def solve_sylvester(a, b, q):
    device = _preferred_linalg_device(a, b, q)
    a = _as_linalg_tensor(a)
    b = _as_linalg_tensor(b)
    q = _as_linalg_tensor(q)
    if device is not None:
        a = a.to(device=device)
        b = b.to(device=device)
        q = q.to(device=device)
    common_dtype = _common_linalg_dtype(a, b, q)
    a = a.to(dtype=common_dtype)
    b = b.to(dtype=common_dtype)
    q = q.to(dtype=common_dtype)

    empty_result = _empty_sylvester_batch_result(a, b, q)
    if empty_result is not None:
        return empty_result

    is_shared_factor = a.shape == b.shape and _torch.allclose(
        a, b, atol=1e-6, rtol=1e-6
    )
    is_shared_hermitian_factor = is_shared_factor and _torch.all(
        _torch.abs(a - a.transpose(-2, -1).conj()) < 1e-6
    )
    if is_shared_hermitian_factor:
        eigvals, eigvecs = eigh(a)
        if _torch.all(eigvals >= 1e-6):
            adjoint_eigvecs = eigvecs.transpose(-2, -1).conj()
            tilde_q = adjoint_eigvecs @ q @ eigvecs
            tilde_x = tilde_q / (eigvals[..., :, None] + eigvals[..., None, :])
            candidate = eigvecs @ tilde_x @ adjoint_eigvecs
            if _sylvester_candidate_is_accurate(a, b, q, candidate):
                return candidate

    is_real_shared_symmetric_factor = (
        is_shared_factor
        and not is_complex(a)
        and _torch.all(_torch.abs(a - a.transpose(-2, -1)) < 1e-6)
    )
    if is_real_shared_symmetric_factor:
        eigvals, eigvecs = eigh(a)
        conditions = _torch.all(eigvals >= 1e-6) or (
            a.shape[-1] >= 2.0
            and _torch.all(eigvals[..., 0] > -1e-6)
            and _torch.all(eigvals[..., 1] >= 1e-6)
            and _torch.all(_torch.abs(q + q.transpose(-2, -1)) < 1e-6)
        )
        if conditions:
            tilde_q = eigvecs.transpose(-2, -1) @ q @ eigvecs
            denominators = eigvals[..., :, None] + eigvals[..., None, :]
            safe_denominators = _torch.where(
                _torch.abs(denominators) < 1e-12,
                _torch.ones((), dtype=denominators.dtype, device=denominators.device),
                denominators,
            )
            tilde_x = tilde_q / safe_denominators
            tilde_x = _torch.where(
                _torch.abs(denominators) < 1e-12,
                _torch.zeros((), dtype=tilde_x.dtype, device=tilde_x.device),
                tilde_x,
            )
            candidate = eigvecs @ tilde_x @ eigvecs.transpose(-2, -1)
            if _sylvester_candidate_is_accurate(a, b, q, candidate):
                return candidate

    solution = _np.vectorize(
        _scipy.linalg.solve_sylvester, signature="(m,m),(n,n),(m,n)->(m,n)"
    )(_as_numpy_no_grad(a), _as_numpy_no_grad(b), _as_numpy_no_grad(q))
    return _torch_as_like(solution, q)


# (TODO) (sait) _torch.linalg.cholesky_ex for even faster way
def is_single_matrix_pd(mat):
    """Check if 2D square matrix is positive definite."""
    mat = _as_linalg_tensor(mat)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return False
    if not bool(_torch.all(_torch.isfinite(mat))):
        return False
    if mat.dtype in [_torch.complex64, _torch.complex128]:
        is_hermitian = bool(
            _torch.all(
                _torch.abs(mat - _torch.conj(_torch.transpose(mat, 0, 1))) < atol
            )
        )
        if not is_hermitian:
            return False
        eigvals = _torch.linalg.eigvalsh(mat)
        return bool(_torch.min(_torch.real(eigvals)) > 0)
    if not bool(_torch.all(_torch.abs(mat - mat.transpose(-2, -1)) < atol)):
        return False
    try:
        _torch.linalg.cholesky(mat)
        return True
    except RuntimeError:
        return False


def fractional_matrix_power(A, t):
    """Compute the fractional power of a matrix."""
    A = _as_linalg_tensor(A)
    A_np = _as_numpy_no_grad(A)
    exponent = _as_numpy_no_grad(t)
    if exponent.ndim != 0:
        raise TypeError("t must be a scalar")
    exponent = exponent.item()

    empty_result = _empty_square_matrix_batch_result(A, "fractional_matrix_power")
    if empty_result is not None:
        return empty_result

    out = _np.vectorize(
        lambda one_matrix: _scipy.linalg.fractional_matrix_power(one_matrix, exponent),
        signature="(n,n)->(n,n)",
    )(A_np)

    if out.dtype.kind == "c":
        target_complex_dtype = _COMPLEX_DTYPE_FOR_TENSOR_DTYPE.get(A.dtype)
        if target_complex_dtype is not None:
            out = out.astype(target_complex_dtype, copy=False)

    return _torch_as_like(out, A)


def polar(a, side="right"):
    """Polar decomposition of a square or rectangular matrix."""
    if side not in {"right", "left"}:
        raise ValueError("`side` must be either 'right' or 'left'")

    a = _as_linalg_tensor(a)
    if _has_empty_matrix_batch(a):
        factor_dim = a.shape[-2] if side == "left" else a.shape[-1]
        factor_shape = a.shape[:-2] + (factor_dim, factor_dim)
        return a.new_empty(a.shape), a.new_empty(factor_shape)

    signature = "(m,n)->(m,n),(m,m)" if side == "left" else "(m,n)->(m,n),(n,n)"
    func = _np.vectorize(_scipy.linalg.polar, signature=signature, excluded=["side"])
    unitary, hermitian = func(_as_numpy_no_grad(a), side=side)

    return _torch_as_like(unitary, a), _torch_as_like(hermitian, a)
