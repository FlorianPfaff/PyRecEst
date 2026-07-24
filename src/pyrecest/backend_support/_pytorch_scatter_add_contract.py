"""Compatibility hook for PyTorch ``scatter_add`` backend semantics."""

from __future__ import annotations

from operator import index as _operator_index


_SCATTER_INDEX_RANGE_ERROR = (
    "scatter_add indices must fit in signed 64-bit integers"
)


def _normalize_pytorch_scatter_dim(dim, torch_module) -> int:
    """Return one PyTorch scatter dimension while rejecting booleans."""
    if isinstance(dim, bool):
        raise TypeError("dim must be an integer")
    if torch_module.is_tensor(dim):
        if dim.ndim != 0 or dim.dtype == torch_module.bool:
            raise TypeError("dim must be an integer")
        dim = dim.item()
    try:
        return _operator_index(dim)
    except TypeError as exc:
        raise TypeError("dim must be an integer") from exc


def _is_integer_scatter_index_dtype(dtype, numpy_module) -> bool:
    """Return whether ``dtype`` can safely represent scatter indices."""

    return not numpy_module.issubdtype(
        dtype,
        numpy_module.bool_,
    ) and numpy_module.issubdtype(dtype, numpy_module.integer)


def _pytorch_scatter_index(index, numpy_module, torch_module, *, device):
    """Return scatter indices as a long tensor without truncating bad inputs."""

    if torch_module.is_tensor(index):
        if (
            index.dtype == torch_module.bool
            or index.dtype.is_floating_point
            or index.dtype.is_complex
        ):
            raise TypeError("scatter_add indices must be integers")
        uint64_dtype = getattr(torch_module, "uint64", None)
        if (
            uint64_dtype is not None
            and index.dtype == uint64_dtype
            and index.device.type != "meta"
        ):
            signed_index = index.to(dtype=torch_module.long)
            if bool(torch_module.any(signed_index < 0).item()):
                raise ValueError(_SCATTER_INDEX_RANGE_ERROR)
        return index.to(device=device, dtype=torch_module.long)

    index_array = numpy_module.asarray(index)
    if isinstance(index, numpy_module.ndarray) or index_array.size:
        if not _is_integer_scatter_index_dtype(index_array.dtype, numpy_module):
            raise TypeError("scatter_add indices must be integers")
        if (
            numpy_module.issubdtype(
                index_array.dtype,
                numpy_module.unsignedinteger,
            )
            and numpy_module.any(
                index_array > numpy_module.iinfo(numpy_module.int64).max
            )
        ):
            raise ValueError(_SCATTER_INDEX_RANGE_ERROR)
    return torch_module.as_tensor(index_array, dtype=torch_module.long, device=device)


def patch_pytorch_scatter_add_contract() -> None:
    """Make raw/public PyTorch ``scatter_add`` accept array-like operands."""
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    original_scatter_add = getattr(raw_pytorch, "scatter_add", None)
    if original_scatter_add is None:
        return
    if getattr(original_scatter_add, "_pyrecest_numpy_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.scatter_add = original_scatter_add
        return

    def scatter_add(input, dim, index, src):  # pylint: disable=redefined-builtin
        values = raw_pytorch.array(input)
        dim_value = _normalize_pytorch_scatter_dim(dim, torch)
        index = _pytorch_scatter_index(index, np, torch, device=values.device)
        if not torch.is_tensor(src):
            src = torch.as_tensor(src, dtype=values.dtype, device=values.device)
        else:
            src = src.to(device=values.device, dtype=values.dtype)
        return torch.scatter_add(values, dim_value, index, src)

    scatter_add.__name__ = getattr(original_scatter_add, "__name__", "scatter_add")
    scatter_add.__doc__ = getattr(original_scatter_add, "__doc__", None)
    scatter_add._pyrecest_arraylike_contract = True
    scatter_add._pyrecest_numpy_contract = True
    raw_pytorch.scatter_add = scatter_add
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.scatter_add = scatter_add


__all__ = ["patch_pytorch_scatter_add_contract"]
