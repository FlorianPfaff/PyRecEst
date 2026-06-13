"""Pytorch based linear algebra backend."""

import numpy as _np
import scipy as _scipy
import torch as _torch
from torch import block_diag  # For PyRecEst
from torch.linalg import pinv  # For PyRecEst
from torch.linalg import (
    cholesky,
    det,
    eig,
    eigh,
    eigvalsh,
    inv,
)
from torch.linalg import matrix_exp as expm
from torch.linalg import (
    matrix_power,
    qr,
    solve,
)

from .._backend_config import np_atol as atol
from ..numpy import linalg as _gsnplinalg
from ._common import array, cast
from ._dtype import (
    get_default_dtype,
    is_complex,
    is_floating,
)


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


def _common_linalg_dtype(*tensors):
    """Return a common floating/complex dtype for torch.linalg operations."""
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = _torch.promote_types(dtype, tensor.dtype)
    if dtype.is_floating_point or dtype.is_complex:
        return dtype
    return _default_linalg_dtype()


class _Logm(_torch.autograd.Function):
    """Torch autograd function for matrix logarithm.

    Implementation based on:
    https://github.com/pytorch/pytorch/issues/9983#issuecomment-891777620
    """

    @staticmethod
    def _logm(x):
        mat_log = _gsnplinalg.logm(_as_numpy_no_grad(x))
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


logm = _Logm.apply


def sqrtm(x):
    x_np = _as_numpy_no_grad(x)
    np_sqrtm = _np.vectorize(_scipy.linalg.sqrtm, signature="(n,m)->(n,m)")(x_np)
    if np_sqrtm.dtype.kind == "c":
        target_complex_dtype = (
            _COMPLEX_DTYPE_FOR_TENSOR_DTYPE.get(x.dtype)
            if isinstance(x, _torch.Tensor)
            else None
        )
        if target_complex_dtype is not None:
            np_sqrtm = np_sqrtm.astype(target_complex_dtype, copy=False)

    return _torch_as_like(np_sqrtm, x)


def svd(x, full_matrices=True, compute_uv=True):
    x = _as_linalg_tensor(x)
    if compute_uv:
        return _torch.linalg.svd(x, full_matrices=full_matrices)

    return _torch.linalg.svdvals(x)


def norm(x, ord=None, axis=None, keepdims=False):
    x = _as_linalg_tensor(x)
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
    return _torch.linalg.matrix_rank(a, atol=atol, rtol=rtol, hermitian=hermitian)


def quadratic_assignment(a, b, options):
    return list(
        _scipy.optimize.quadratic_assignment(
            _as_numpy_no_grad(a), _as_numpy_no_grad(b), options=options
        ).col_ind
    )


def solve_sylvester(a, b, q):
    a = _as_linalg_tensor(a)
    b = _as_linalg_tensor(b)
    q = _as_linalg_tensor(q)
    common_dtype = _common_linalg_dtype(a, b, q)
    a = a.to(dtype=common_dtype)
    b = b.to(dtype=common_dtype)
    q = q.to(dtype=common_dtype)
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
            return eigvecs @ tilde_x @ adjoint_eigvecs

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
            tilde_x = tilde_q / (
                eigvals[..., :, None]
                + eigvals[..., None, :]
                + _torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
            )
            return eigvecs @ tilde_x @ eigvecs.transpose(-2, -1)

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
    if mat.dtype in [_torch.complex64, _torch.complex128]:
        is_hermitian = _torch.all(
            _torch.abs(mat - _torch.conj(_torch.transpose(mat, 0, 1))) < atol
        )
        if not is_hermitian:
            return False
        eigvals = _torch.linalg.eigvalsh(mat)
        return _torch.min(_torch.real(eigvals)) > 0
    if not _torch.all(_torch.abs(mat - mat.transpose(-2, -1)) < atol):
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
    out = _np.vectorize(
        lambda one_matrix: _scipy.linalg.fractional_matrix_power(one_matrix, t),
        signature="(n,n)->(n,n)",
    )(A_np)

    if out.dtype.kind == "c":
        target_complex_dtype = _COMPLEX_DTYPE_FOR_TENSOR_DTYPE.get(A.dtype)
        if target_complex_dtype is not None:
            out = out.astype(target_complex_dtype, copy=False)

    return _torch_as_like(out, A)


def polar(a, side="right"):
    """Polar decomposition of a square or rectangular matrix."""
    a = _as_linalg_tensor(a)
    signature = "(m,n)->(m,n),(m,m)" if side == "left" else "(m,n)->(m,n),(n,n)"
    func = _np.vectorize(_scipy.linalg.polar, signature=signature, excluded=["side"])
    unitary, hermitian = func(_as_numpy_no_grad(a), side=side)

    return _torch_as_like(unitary, a), _torch_as_like(hermitian, a)
