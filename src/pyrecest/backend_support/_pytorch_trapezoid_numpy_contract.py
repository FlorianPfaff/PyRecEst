"""PyTorch ``trapezoid`` NumPy compatibility hook."""

from __future__ import annotations

from operator import index as _operator_index


def _preferred_pytorch_device(torch_module, *values):
    """Return an existing non-CPU tensor device, falling back to any tensor."""
    for value in values:
        if torch_module.is_tensor(value) and value.device.type != "cpu":
            return value.device
    for value in values:
        if torch_module.is_tensor(value):
            return value.device
    return None


def _trapezoid_axis(axis) -> int:
    """Return a NumPy-style integer axis while rejecting boolean axes."""
    if isinstance(axis, bool) or type(axis).__name__ == "bool_":
        raise TypeError("axis must be an integer")
    try:
        return _operator_index(axis)
    except TypeError as exc:
        raise TypeError("axis must be an integer") from exc


def _as_trapezoid_tensor(value, torch_module, *, device=None, dtype=None):
    """Coerce one trapezoid argument without moving existing tensors unnecessarily."""
    if torch_module.is_tensor(value):
        target_device = device if device is not None else value.device
        target_dtype = dtype if dtype is not None else value.dtype
        if value.device != target_device or value.dtype != target_dtype:
            return value.to(device=target_device, dtype=target_dtype)
        return value
    return torch_module.as_tensor(value, device=device, dtype=dtype)


def _promote_trapezoid_tensor(value, raw_pytorch):
    """Promote integer and boolean inputs before PyTorch integration."""
    if value.dtype.is_floating_point or value.dtype.is_complex:
        return value
    return value.to(dtype=raw_pytorch.get_default_dtype())


def patch_pytorch_trapezoid_numpy_contract() -> None:
    """Patch raw/public PyTorch ``trapezoid`` to accept NumPy-style inputs."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    original_trapezoid = getattr(raw_pytorch, "trapezoid", None)
    if original_trapezoid is None:
        return
    if getattr(original_trapezoid, "_pyrecest_numpy_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.trapezoid = original_trapezoid
        return

    def trapezoid(y, x=None, dx=1.0, axis=-1):
        dim = _trapezoid_axis(axis)
        device = _preferred_pytorch_device(torch, y, x)
        y = _as_trapezoid_tensor(y, torch, device=device)

        if x is None:
            y = _promote_trapezoid_tensor(y, raw_pytorch)
            return torch.trapezoid(y, dx=dx, dim=dim)

        x = _as_trapezoid_tensor(x, torch, device=y.device)
        result_dtype = torch.promote_types(y.dtype, x.dtype)
        if not (result_dtype.is_floating_point or result_dtype.is_complex):
            result_dtype = raw_pytorch.get_default_dtype()
        y = y.to(dtype=result_dtype)
        x = x.to(dtype=result_dtype)
        return torch.trapezoid(y, x=x, dim=dim)

    trapezoid.__name__ = getattr(original_trapezoid, "__name__", "trapezoid")
    trapezoid.__doc__ = getattr(original_trapezoid, "__doc__", None)
    trapezoid._pyrecest_numpy_contract = True
    raw_pytorch.trapezoid = trapezoid
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.trapezoid = trapezoid


def _patch_rectangular_pytorch_triangular_vector_contract() -> None:
    """Patch PyTorch triangular vector helpers for rectangular matrices."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"

    def _make_triangular_to_vec(helper_name, torch_index_helper, original_helper):
        def triangular_to_vec(x, k=0):
            x = raw_pytorch.array(x)
            if x.ndim < 2:
                raise ValueError(
                    "triangular vector helpers require at least two dimensions"
                )
            rows, cols = torch_index_helper(
                row=x.shape[-2],
                col=x.shape[-1],
                offset=_operator_index(k),
                device=x.device,
            )
            return x[..., rows, cols]

        triangular_to_vec.__name__ = getattr(original_helper, "__name__", helper_name)
        triangular_to_vec.__doc__ = getattr(original_helper, "__doc__", None)
        triangular_to_vec._pyrecest_numpy_contract = True
        triangular_to_vec._pyrecest_arraylike_contract = True
        return triangular_to_vec

    for helper_name, torch_index_helper in (
        ("tril_to_vec", torch.tril_indices),
        ("triu_to_vec", torch.triu_indices),
    ):
        original_helper = getattr(raw_pytorch, helper_name, None)
        if original_helper is None:
            continue
        helper = _make_triangular_to_vec(
            helper_name, torch_index_helper, original_helper
        )
        setattr(raw_pytorch, helper_name, helper)
        if active_pytorch_backend:
            setattr(backend, helper_name, helper)


def _patch_rectangular_jax_triangular_vector_contract() -> None:
    """Patch JAX triangular vector helpers for rectangular matrices."""
    try:
        import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.jax as raw_jax  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - JAX backend may be unavailable
        return

    active_jax_backend = getattr(backend, "__backend_name__", None) == "jax"

    def _make_triangular_to_vec(helper_name, jax_index_helper, original_helper):
        def triangular_to_vec(x, k=0):
            x = jnp.asarray(x)
            if x.ndim < 2:
                raise ValueError(
                    "triangular vector helpers require at least two dimensions"
                )
            rows, cols = jax_index_helper(
                n=x.shape[-2],
                k=_operator_index(k),
                m=x.shape[-1],
            )
            return x[..., rows, cols]

        triangular_to_vec.__name__ = getattr(original_helper, "__name__", helper_name)
        triangular_to_vec.__doc__ = getattr(original_helper, "__doc__", None)
        triangular_to_vec._pyrecest_numpy_contract = True
        triangular_to_vec._pyrecest_arraylike_contract = True
        return triangular_to_vec

    for helper_name, jax_index_helper in (
        ("tril_to_vec", jnp.tril_indices),
        ("triu_to_vec", jnp.triu_indices),
    ):
        original_helper = getattr(raw_jax, helper_name, None)
        if original_helper is None:
            continue
        helper = _make_triangular_to_vec(helper_name, jax_index_helper, original_helper)
        setattr(raw_jax, helper_name, helper)
        if active_jax_backend:
            setattr(backend, helper_name, helper)


_patch_rectangular_pytorch_triangular_vector_contract()
_patch_rectangular_jax_triangular_vector_contract()
