"""Runtime patch for PyTorch ``take`` index-array validation."""

from __future__ import annotations

import numpy as np

_TAKE_INDEX_TYPE_MESSAGE = (
    "indices must be integers or boolean values when supplied as an array"
)


def _validate_take_indices(indices, torch_module):
    """Reject array index dtypes that NumPy cannot cast with ``same_kind``."""

    if torch_module.is_tensor(indices):
        if indices.dtype.is_floating_point or indices.dtype.is_complex:
            raise TypeError(_TAKE_INDEX_TYPE_MESSAGE)
        return indices

    if isinstance(indices, np.ndarray) and not np.can_cast(
        indices.dtype,
        np.dtype(np.intp),
        casting="same_kind",
    ):
        raise TypeError(_TAKE_INDEX_TYPE_MESSAGE)
    return indices


def patch_pytorch_take_index_contract() -> None:
    """Make public and raw PyTorch ``take`` reject invalid index arrays."""

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch may be unavailable
        return

    original_take = getattr(raw_pytorch, "take", None)
    if original_take is None:
        return
    if getattr(original_take, "_pyrecest_take_index_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.take = original_take
        return

    def take(a, indices, axis=None, out=None, mode=None):
        indices = _validate_take_indices(indices, torch)
        return original_take(a, indices, axis=axis, out=out, mode=mode)

    take.__name__ = getattr(original_take, "__name__", "take")
    take.__doc__ = getattr(original_take, "__doc__", None)
    take._pyrecest_take_index_contract = True
    raw_pytorch.take = take
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.take = take
