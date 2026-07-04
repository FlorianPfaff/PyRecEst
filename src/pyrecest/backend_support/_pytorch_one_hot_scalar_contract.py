"""PyTorch ``one_hot`` scalar-label compatibility hook."""

from __future__ import annotations

from operator import index as _operator_index


def _patch_pytorch_one_hot_scalar_contract() -> None:
    """Patch raw/public PyTorch ``one_hot`` to handle scalar labels correctly."""
    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch as torch_module  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    original_one_hot = getattr(pytorch_backend, "one_hot", None)
    if original_one_hot is None:
        return
    if getattr(original_one_hot, "_pyrecest_scalar_label_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.one_hot = original_one_hot
        return

    def one_hot(labels, num_classes):
        num_classes = _operator_index(num_classes)
        if torch_module.is_tensor(labels):
            if (
                labels.dtype == torch_module.bool
                or labels.dtype.is_floating_point
                or labels.dtype.is_complex
            ):
                return original_one_hot(labels, num_classes)
            labels = labels.to(dtype=torch_module.long)
        else:
            labels = torch_module.as_tensor(labels, dtype=torch_module.long)
        return torch_module.nn.functional.one_hot(labels, num_classes).to(
            dtype=torch_module.uint8
        )

    one_hot.__name__ = getattr(original_one_hot, "__name__", "one_hot")
    one_hot.__doc__ = getattr(original_one_hot, "__doc__", None)
    one_hot._pyrecest_scalar_label_contract = True
    pytorch_backend.one_hot = one_hot
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.one_hot = one_hot


def _patch_jax_one_hot_bounds_contract() -> None:
    """Patch raw/public JAX ``one_hot`` to mask labels outside the class range."""
    try:
        import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.jax as jax_backend  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - JAX backend may be unavailable
        return

    original_one_hot = getattr(jax_backend, "one_hot", None)
    if original_one_hot is None:
        return
    if getattr(original_one_hot, "_pyrecest_label_bounds_contract", False):
        if getattr(backend, "__backend_name__", None) == "jax":
            backend.one_hot = original_one_hot
        return

    def _uint8_one_hot(labels, num_classes):
        num_classes = _operator_index(num_classes)
        labels = jnp.asarray(labels)
        if labels.dtype.kind not in "iu":
            raise TypeError("one_hot labels must be integer-valued")

        valid = (labels >= 0) & (labels < num_classes)
        safe_labels = jnp.where(valid, labels, 0)
        encoded = jnp.eye(num_classes, dtype=jnp.uint8)[safe_labels]
        return jnp.where(valid[..., None], encoded, jnp.zeros_like(encoded))

    def one_hot(labels=None, num_classes=None, *, indices=None, depth=None):
        if indices is not None:
            if labels is not None:
                raise TypeError("one_hot() got both 'labels' and 'indices'")
            labels = indices
        if depth is not None:
            if num_classes is not None and num_classes != depth:
                raise TypeError("one_hot() got both 'num_classes' and 'depth'")
            num_classes = depth
        if labels is None:
            raise TypeError("one_hot() missing required argument 'labels'")
        if num_classes is None:
            raise TypeError("one_hot() missing required argument 'num_classes'")
        return _uint8_one_hot(labels, num_classes)

    one_hot.__name__ = getattr(original_one_hot, "__name__", "one_hot")
    one_hot.__doc__ = getattr(original_one_hot, "__doc__", None)
    one_hot._pyrecest_backend_contract = True
    one_hot._pyrecest_label_bounds_contract = True
    jax_backend.one_hot = one_hot
    if getattr(backend, "__backend_name__", None) == "jax":
        backend.one_hot = one_hot


def patch_pytorch_one_hot_scalar_contract() -> None:
    """Patch one_hot compatibility contracts used by runtime stability hooks."""
    _patch_pytorch_one_hot_scalar_contract()
    _patch_jax_one_hot_bounds_contract()


__all__ = ["patch_pytorch_one_hot_scalar_contract"]
