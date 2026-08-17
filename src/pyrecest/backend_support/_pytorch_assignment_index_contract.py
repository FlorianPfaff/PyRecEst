"""Runtime patch for PyTorch assignment index dtype validation."""

from __future__ import annotations

_ASSIGNMENT_INDEX_TYPE_MESSAGE = (
    "arrays used as indices must be of integer (or boolean) type"
)


def _is_boolean_assignment_index(index, torch_module) -> bool:
    """Return whether one non-tuple index is a homogeneous boolean mask."""

    if isinstance(index, tuple):
        # Tuples represent multidimensional index components rather than one mask.
        return False
    if torch_module.is_tensor(index):
        return index.dtype == torch_module.bool
    try:
        index_tensor = torch_module.as_tensor(index)
    except (TypeError, ValueError, RuntimeError):
        return False
    return index_tensor.dtype == torch_module.bool


def _integer_dtypes(torch_module):
    """Return integer dtypes available in the installed PyTorch version."""

    return {
        getattr(torch_module, dtype_name)
        for dtype_name in (
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "int8",
            "int16",
            "int32",
            "int64",
        )
        if hasattr(torch_module, dtype_name)
    }


def _as_assignment_index(index, *, device, torch_module):
    """Convert an assignment index without silently truncating its dtype."""

    if torch_module.is_tensor(index):
        index_tensor = index.to(device=device)
    else:
        try:
            index_tensor = torch_module.as_tensor(index, device=device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise IndexError(_ASSIGNMENT_INDEX_TYPE_MESSAGE) from exc

    if index_tensor.dtype == torch_module.bool:
        return index_tensor
    if index_tensor.dtype not in _integer_dtypes(torch_module):
        raise IndexError(_ASSIGNMENT_INDEX_TYPE_MESSAGE)
    return index_tensor.to(dtype=torch_module.long)


def patch_pytorch_assignment_index_contract() -> None:
    """Make PyTorch assignment indices follow integer/boolean dtype semantics."""

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch may be unavailable
        return

    current_is_boolean = getattr(raw_pytorch, "_is_boolean", None)
    current_as_assignment_index = getattr(raw_pytorch, "_as_assignment_index", None)
    if getattr(
        current_is_boolean,
        "_pyrecest_assignment_index_dtype_contract",
        False,
    ) and getattr(
        current_as_assignment_index,
        "_pyrecest_assignment_index_dtype_contract",
        False,
    ):
        return

    def _is_boolean(index):
        return _is_boolean_assignment_index(index, torch)

    def _as_index(index, *, device):
        return _as_assignment_index(index, device=device, torch_module=torch)

    for helper in (_is_boolean, _as_index):
        helper._pyrecest_assignment_index_dtype_contract = True
        # The older uint8 patch checks this marker. Keep it set so a later
        # compatibility pass cannot replace the stricter helpers again.
        helper._pyrecest_uint8_assignment_index_contract = True

    raw_pytorch._is_boolean = _is_boolean
    raw_pytorch._as_assignment_index = _as_index
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.assignment = raw_pytorch.assignment
        backend.assignment_by_sum = raw_pytorch.assignment_by_sum
