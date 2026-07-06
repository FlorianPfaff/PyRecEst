"""PyTorch broadcast-shape compatibility hook."""

from __future__ import annotations


def patch_pytorch_broadcast_shape_entry_contract() -> None:
    """Keep tuple/list broadcast dimensions aligned with the existing contract."""

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover
        return

    original_broadcast_to = getattr(pytorch_backend, "broadcast_to", None)
    if original_broadcast_to is None:
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    if getattr(original_broadcast_to, "_pyrecest_shape_entry_contract", False):
        if active_pytorch_backend:
            backend.broadcast_to = original_broadcast_to
        return

    def _has_rejected_shape_entry(shape):
        if torch.is_tensor(shape):
            return False
        if not isinstance(shape, (list, tuple)):
            return False
        return any(
            torch.is_tensor(dimension) and dimension.dtype == torch.bool
            for dimension in shape
        )

    def broadcast_to(x, shape):
        if _has_rejected_shape_entry(shape):
            raise TypeError("broadcast shape entries must be integers")
        return original_broadcast_to(x, shape)

    broadcast_to.__name__ = getattr(original_broadcast_to, "__name__", "broadcast_to")
    broadcast_to.__doc__ = getattr(original_broadcast_to, "__doc__", None)
    broadcast_to._pyrecest_numpy_contract = getattr(
        original_broadcast_to,
        "_pyrecest_numpy_contract",
        False,
    )
    broadcast_to._pyrecest_shape_entry_contract = True
    pytorch_backend.broadcast_to = broadcast_to
    if active_pytorch_backend:
        backend.broadcast_to = broadcast_to
