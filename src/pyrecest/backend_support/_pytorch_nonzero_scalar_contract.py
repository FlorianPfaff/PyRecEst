"""Runtime patch for PyTorch ``nonzero`` scalar validation."""

from __future__ import annotations

_NONZERO_SCALAR_MESSAGE = (
    "Calling nonzero on 0d arrays is not allowed. "
    "Use np.atleast_1d(scalar).nonzero() instead."
)


def patch_pytorch_nonzero_scalar_contract() -> None:
    """Make public and raw PyTorch ``nonzero`` reject 0-D inputs."""

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch may be unavailable
        return

    original_nonzero = getattr(raw_pytorch, "nonzero", None)
    if original_nonzero is None:
        return
    if getattr(original_nonzero, "_pyrecest_nonzero_scalar_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.nonzero = original_nonzero
        return

    def nonzero(x):
        values = x if torch.is_tensor(x) else torch.as_tensor(x)
        if values.ndim == 0:
            raise ValueError(_NONZERO_SCALAR_MESSAGE)
        return original_nonzero(values)

    nonzero.__name__ = getattr(original_nonzero, "__name__", "nonzero")
    nonzero.__doc__ = getattr(original_nonzero, "__doc__", None)
    nonzero._pyrecest_nonzero_scalar_contract = True
    raw_pytorch.nonzero = nonzero
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.nonzero = nonzero


__all__ = ["patch_pytorch_nonzero_scalar_contract"]
