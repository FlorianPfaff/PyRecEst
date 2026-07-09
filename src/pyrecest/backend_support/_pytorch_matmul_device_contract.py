"""PyTorch ``matmul``/``matvec`` device compatibility hook."""

from __future__ import annotations


def _preferred_pytorch_device(torch_module, *values):
    """Return an existing non-CPU tensor device, falling back to any tensor."""
    for value in values:
        if torch_module.is_tensor(value) and value.device.type != "cpu":
            return value.device
    for value in values:
        if torch_module.is_tensor(value):
            return value.device
    return None


def _promoted_pair(raw_pytorch, torch_module, left, right, *, device):
    """Return operands on a common dtype and the selected device."""
    left = raw_pytorch.array(left)
    right = raw_pytorch.array(right)
    dtype = torch_module.promote_types(left.dtype, right.dtype)

    if device is not None:
        return left.to(device=device, dtype=dtype), right.to(device=device, dtype=dtype)
    return left.to(dtype=dtype), right.to(dtype=dtype)


def patch_pytorch_matmul_device_contract() -> None:
    """Patch raw/public PyTorch ``matmul`` and ``matvec`` device handling."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    original_matmul = getattr(raw_pytorch, "matmul", None)
    original_matvec = getattr(raw_pytorch, "matvec", None)
    if original_matmul is None or original_matvec is None:
        return

    helper_names = ("matmul", "matvec")
    if all(
        getattr(
            getattr(raw_pytorch, helper_name, None),
            "_pyrecest_matmul_device_contract",
            False,
        )
        for helper_name in helper_names
    ):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            for helper_name in helper_names:
                setattr(backend, helper_name, getattr(raw_pytorch, helper_name))
        return

    def matmul(x, y, out=None):
        device = _preferred_pytorch_device(torch, x, y, out)
        x, y = _promoted_pair(raw_pytorch, torch, x, y, device=device)

        if out is not None:
            return torch.matmul(x, y, out=out)
        return torch.matmul(x, y)

    def matvec(A, b):
        device = _preferred_pytorch_device(torch, A, b)
        A, b = _promoted_pair(raw_pytorch, torch, A, b, device=device)

        if A.ndim == 2 and b.ndim == 1:
            return torch.mv(A, b)

        if b.ndim == 1:  # A.ndim > 2
            return torch.matmul(A, b)

        if A.ndim == 2:  # b.ndim > 1
            return torch.einsum("ij,...j->...i", A, b)

        return torch.einsum("...ij,...j->...i", A, b)

    for helper_name, helper, original_helper in (
        ("matmul", matmul, original_matmul),
        ("matvec", matvec, original_matvec),
    ):
        helper.__name__ = getattr(original_helper, "__name__", helper_name)
        helper.__doc__ = getattr(original_helper, "__doc__", None)
        helper._pyrecest_matmul_device_contract = True
        helper._pyrecest_device_contract = True
        helper._pyrecest_numpy_contract = True
        setattr(raw_pytorch, helper_name, helper)
        if getattr(backend, "__backend_name__", None) == "pytorch":
            setattr(backend, helper_name, helper)
