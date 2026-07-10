"""Runtime patch for PyTorch ``split`` cut-index validation."""

from __future__ import annotations

from operator import index as _operator_index

import numpy as np
from pyrecest.backend_support._pytorch_take_index_contract import (
    patch_pytorch_take_index_contract as _patch_pytorch_take_index_contract,
)

_SPLIT_INDEX_TYPE_MESSAGE = (
    "slice indices must be integers or None or have an __index__ method"
)


def _normalize_split_cut_indices(indices_or_sections, torch_module):
    """Return integer cut points without silently truncating invalid values."""

    if isinstance(indices_or_sections, (str, bytes)):
        raise TypeError(_SPLIT_INDEX_TYPE_MESSAGE)

    if torch_module.is_tensor(indices_or_sections):
        if indices_or_sections.ndim == 0:
            return indices_or_sections
        if indices_or_sections.ndim != 1:
            raise ValueError("indices_or_sections must be a 1-D sequence")
        cut_points = tuple(indices_or_sections)
    else:
        cut_array = np.asarray(indices_or_sections)
        if cut_array.ndim == 0:
            return indices_or_sections
        if cut_array.ndim != 1:
            raise ValueError("indices_or_sections must be a 1-D sequence")
        cut_points = (
            tuple(indices_or_sections)
            if isinstance(indices_or_sections, (list, tuple))
            else tuple(cut_array)
        )

    normalized = []
    for cut_point in cut_points:
        if torch_module.is_tensor(cut_point) and cut_point.dtype == torch_module.bool:
            raise TypeError(_SPLIT_INDEX_TYPE_MESSAGE)
        try:
            normalized.append(_operator_index(cut_point))
        except TypeError as exc:
            raise TypeError(_SPLIT_INDEX_TYPE_MESSAGE) from exc
    return tuple(normalized)


def patch_pytorch_split_index_contract() -> None:
    """Make public and raw PyTorch ``split`` reject non-integer cut points."""

    _patch_pytorch_take_index_contract()

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch may be unavailable
        return

    original_split = getattr(raw_pytorch, "split", None)
    if original_split is None:
        return
    if getattr(original_split, "_pyrecest_split_index_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.split = original_split
        return

    def split(x, indices_or_sections, axis=0):
        normalized = _normalize_split_cut_indices(indices_or_sections, torch)
        return original_split(x, normalized, axis=axis)

    split.__name__ = getattr(original_split, "__name__", "split")
    split.__doc__ = getattr(original_split, "__doc__", None)
    split._pyrecest_split_index_contract = True
    raw_pytorch.split = split
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.split = split
