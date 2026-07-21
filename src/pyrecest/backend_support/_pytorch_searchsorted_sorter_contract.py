"""Runtime validation for PyTorch ``searchsorted`` sorter arrays."""

from __future__ import annotations

from pyrecest._backend_runtime_patches import (
    patch_pytorch_searchsorted_contract as _patch_pytorch_searchsorted_contract,
)

_SORTER_TYPE_MESSAGE = "sorter must only contain integers"
_SORTER_SHAPE_MESSAGE = "could not parse sorter argument"


def _validate_sorter(sorter, numpy_module, torch_module) -> None:
    """Reject sorter values that NumPy does not accept as integer indices."""

    if sorter is None:
        return

    if torch_module.is_tensor(sorter):
        if sorter.ndim != 1:
            raise TypeError(_SORTER_SHAPE_MESSAGE)
        if (
            sorter.dtype == torch_module.bool
            or sorter.dtype.is_floating_point
            or sorter.dtype.is_complex
            or str(sorter.dtype) == "torch.uint64"
        ):
            raise TypeError(_SORTER_TYPE_MESSAGE)
        return

    try:
        sorter_array = numpy_module.asarray(sorter)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TypeError(_SORTER_SHAPE_MESSAGE) from exc

    if sorter_array.ndim != 1:
        raise TypeError(_SORTER_SHAPE_MESSAGE)
    if numpy_module.issubdtype(
        sorter_array.dtype, numpy_module.bool_
    ) or not numpy_module.can_cast(
        sorter_array.dtype,
        numpy_module.dtype(numpy_module.intp),
        casting="safe",
    ):
        raise TypeError(_SORTER_TYPE_MESSAGE)


def patch_pytorch_searchsorted_sorter_contract() -> None:
    """Prevent lossy coercion of PyTorch ``searchsorted`` sorter arrays."""

    _patch_pytorch_searchsorted_contract()

    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch may be unavailable
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    original_searchsorted = getattr(raw_pytorch, "searchsorted", None)
    if original_searchsorted is None:
        return
    if getattr(
        original_searchsorted,
        "_pyrecest_searchsorted_sorter_contract",
        False,
    ):
        if active_pytorch_backend:
            backend.searchsorted = original_searchsorted
        return

    def searchsorted(
        a,
        v,
        side="left",
        sorter=None,
        *,
        out=None,
        right=False,
        out_int32=False,
    ):
        _validate_sorter(sorter, np, torch)
        return original_searchsorted(
            a,
            v,
            side=side,
            sorter=sorter,
            out=out,
            right=right,
            out_int32=out_int32,
        )

    searchsorted.__name__ = getattr(
        original_searchsorted,
        "__name__",
        "searchsorted",
    )
    searchsorted.__doc__ = getattr(original_searchsorted, "__doc__", None)
    searchsorted._pyrecest_searchsorted_contract = True
    searchsorted._pyrecest_searchsorted_sorter_contract = True
    raw_pytorch.searchsorted = searchsorted
    if active_pytorch_backend:
        backend.searchsorted = searchsorted
