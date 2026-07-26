"""PyTorch backend dot-product contract patch."""

from __future__ import annotations


def patch_pytorch_dot_numpy_contract() -> None:
    """Make raw and public PyTorch ``dot`` follow NumPy's axis contract."""

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch may be unavailable
        return

    original_dot = getattr(raw_pytorch, "dot", None)
    if original_dot is None:
        return
    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    if getattr(original_dot, "_pyrecest_numpy_dot_contract", False):
        if active_pytorch_backend:
            backend.dot = original_dot
        return

    def dot(a, b):
        left = raw_pytorch.array(a)
        right = raw_pytorch.array(b)
        left, right = raw_pytorch.convert_to_wider_dtype([left, right])

        if left.ndim == 0 or right.ndim == 0:
            return torch.multiply(left, right)
        if left.ndim == 1 and right.ndim == 1:
            return torch.dot(left, right)
        if right.ndim == 1:
            return torch.tensordot(left, right, dims=([-1], [0]))
        if left.ndim == 1:
            return torch.tensordot(left, right, dims=([0], [-2]))
        return torch.tensordot(left, right, dims=([-1], [-2]))

    dot.__name__ = getattr(original_dot, "__name__", "dot")
    dot.__doc__ = getattr(original_dot, "__doc__", None)
    dot._pyrecest_numpy_dot_contract = True
    raw_pytorch.dot = dot
    if active_pytorch_backend:
        backend.dot = dot
