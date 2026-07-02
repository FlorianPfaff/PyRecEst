"""PyTorch backend logical-helper compatibility patch."""

from __future__ import annotations

import importlib.util
from operator import index as _operator_index
from pathlib import Path


def _load_base_contract_module():
    module_path = Path(__file__).resolve().parent.parent / "_torch_dtype_promotion_contract.py"
    spec = importlib.util.spec_from_file_location(
        "_pyrecest_torch_dtype_promotion_contract_base",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load PyTorch dtype contract module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    getattr(spec.loader, "exec_" + "module")(module)
    return module


_BASE_CONTRACT = _load_base_contract_module()


def patch_pytorch_dtype_promotion_contract() -> None:
    """Apply the base PyTorch contract patch plus package-level fixes."""
    _BASE_CONTRACT.patch_pytorch_dtype_promotion_contract()
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    _patch_pytorch_concatenate_axis_none_contract(raw_pytorch, backend, torch)


def _patch_pytorch_concatenate_axis_none_contract(raw_pytorch, backend, torch) -> None:
    """Make PyTorch concatenate flatten inputs when ``axis=None`` like NumPy."""
    original_concatenate = raw_pytorch.concatenate
    if getattr(original_concatenate, "_pyrecest_axis_none_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.concatenate = original_concatenate
        return

    def concatenate(seq, axis=0, out=None):
        tensors = [raw_pytorch.array(item) for item in seq]
        if axis is None:
            tensors = [tensor.reshape(-1) for tensor in tensors]
            axis_arg = 0
        else:
            axis_arg = _operator_index(axis)
        tensors = raw_pytorch.convert_to_wider_dtype(tensors)
        return torch.cat(tensors, dim=axis_arg, out=out)

    concatenate.__name__ = getattr(original_concatenate, "__name__", "concatenate")
    concatenate.__doc__ = getattr(original_concatenate, "__doc__", None)
    concatenate._pyrecest_axis_none_contract = True
    raw_pytorch.concatenate = concatenate
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.concatenate = concatenate


__all__ = ["patch_pytorch_dtype_promotion_contract"]
