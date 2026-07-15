"""PyTorch ``one_hot`` and ``take`` compatibility hooks."""

from __future__ import annotations

from operator import index as _operator_index

_INTEGER_MESSAGE = "num_classes must be an integer"


def _is_boolean_take_axis(axis, torch_module) -> bool:
    """Return whether ``axis`` is a boolean scalar, not an integer axis."""
    if isinstance(axis, bool) or type(axis).__name__ == "bool_":
        return True
    return bool(
        torch_module.is_tensor(axis)
        and axis.ndim == 0
        and axis.dtype == torch_module.bool
    )


def _normalize_num_classes(num_classes, torch_module) -> int:
    """Return a non-boolean integer ``num_classes`` value."""
    if isinstance(num_classes, bool) or type(num_classes).__name__ == "bool_":
        raise TypeError(f"{_INTEGER_MESSAGE}, not boolean")
    if torch_module.is_tensor(num_classes):
        if num_classes.ndim != 0 or num_classes.dtype == torch_module.bool:
            raise TypeError(_INTEGER_MESSAGE)
        num_classes = num_classes.item()
    try:
        return _operator_index(num_classes)
    except TypeError as exc:
        raise TypeError(_INTEGER_MESSAGE) from exc


def _patch_pytorch_take_axis_contract(pytorch_backend, torch_module) -> None:
    """Patch raw/public PyTorch ``take`` to reject non-integer axes."""
    original_normalizer = getattr(pytorch_backend, "_normalize_take_axis", None)
    if original_normalizer is None:
        return
    if getattr(original_normalizer, "_pyrecest_axis_contract", False):
        return

    def _normalize_take_axis(axis, ndim_):
        if axis is None:
            return None
        if _is_boolean_take_axis(axis, torch_module):
            raise TypeError("an integer is required for the axis")
        try:
            axis = _operator_index(axis)
        except TypeError as exc:
            raise TypeError("an integer is required for the axis") from exc
        if axis < 0:
            axis += ndim_
        if axis < 0 or axis >= ndim_:
            raise IndexError(
                f"axis {axis} is out of bounds for array of dimension {ndim_}"
            )
        return axis

    _normalize_take_axis.__name__ = getattr(
        original_normalizer,
        "__name__",
        "_normalize_take_axis",
    )
    _normalize_take_axis.__doc__ = getattr(original_normalizer, "__doc__", None)
    _normalize_take_axis._pyrecest_axis_contract = True
    pytorch_backend._normalize_take_axis = _normalize_take_axis


def _patch_pytorch_one_hot_scalar_contract(
    pytorch_backend,
    backend,
    torch_module,
) -> None:
    """Patch raw/public PyTorch ``one_hot`` to handle scalar labels correctly."""
    original_one_hot = getattr(pytorch_backend, "one_hot", None)
    if original_one_hot is None:
        return
    if getattr(original_one_hot, "_pyrecest_scalar_label_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.one_hot = original_one_hot
        return

    def one_hot(labels, num_classes):
        num_classes = _normalize_num_classes(num_classes, torch_module)
        if not torch_module.is_tensor(labels):
            labels = torch_module.as_tensor(labels)
        if (
            labels.dtype == torch_module.bool
            or labels.dtype.is_floating_point
            or labels.dtype.is_complex
        ):
            return original_one_hot(labels, num_classes)
        labels = labels.to(dtype=torch_module.long)
        return torch_module.nn.functional.one_hot(labels, num_classes).to(
            dtype=torch_module.uint8
        )

    one_hot.__name__ = getattr(original_one_hot, "__name__", "one_hot")
    one_hot.__doc__ = getattr(original_one_hot, "__doc__", None)
    one_hot._pyrecest_scalar_label_contract = True
    pytorch_backend.one_hot = one_hot
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.one_hot = one_hot


def patch_pytorch_one_hot_scalar_contract() -> None:
    """Patch small PyTorch backend compatibility contracts."""
    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch as torch_module  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    _patch_pytorch_one_hot_scalar_contract(
        pytorch_backend,
        backend,
        torch_module,
    )
    _patch_pytorch_take_axis_contract(pytorch_backend, torch_module)


__all__ = ["patch_pytorch_one_hot_scalar_contract"]
