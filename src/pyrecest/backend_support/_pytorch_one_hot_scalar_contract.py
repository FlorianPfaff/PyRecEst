"""PyTorch ``one_hot``, ``take``, and gamma compatibility hooks."""

from __future__ import annotations

from operator import index as _operator_index

_INTEGER_MESSAGE = "num_classes must be an integer"
_NONNEGATIVE_MESSAGE = "num_classes must be non-negative"


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
    """Return a non-negative, non-boolean integer ``num_classes`` value."""
    if isinstance(num_classes, bool) or type(num_classes).__name__ == "bool_":
        raise TypeError(f"{_INTEGER_MESSAGE}, not boolean")
    if torch_module.is_tensor(num_classes):
        if num_classes.ndim != 0 or num_classes.dtype == torch_module.bool:
            raise TypeError(_INTEGER_MESSAGE)
        num_classes = num_classes.item()
    try:
        normalized = _operator_index(num_classes)
    except TypeError as exc:
        raise TypeError(_INTEGER_MESSAGE) from exc
    if normalized < 0:
        raise ValueError(_NONNEGATIVE_MESSAGE)
    return normalized


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


def _patch_pytorch_gamma_autograd_contract(
    pytorch_backend,
    backend,
    torch_module,
) -> None:
    """Keep inactive reflection singularities out of ``gamma`` gradients."""
    original_gamma = getattr(pytorch_backend, "gamma", None)
    if original_gamma is None:
        return
    if getattr(original_gamma, "_pyrecest_finite_gradient_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.gamma = original_gamma
        return

    def gamma(a, out=None):
        values = pytorch_backend.array(a)
        if not pytorch_backend.is_floating(values):
            if pytorch_backend.is_complex(values):
                raise TypeError(
                    "gamma is only supported for real-valued PyTorch inputs"
                )
            values = pytorch_backend.cast(
                values, dtype=pytorch_backend.get_default_dtype()
            )

        positive_branch = torch_module.exp(torch_module.special.gammaln(values))
        negative_mask = values < 0
        reflection_values = torch_module.where(
            negative_mask,
            values,
            torch_module.full_like(values, -0.5),
        )
        reflected_branch = torch_module.pi / (
            torch_module.sin(torch_module.pi * reflection_values)
            * torch_module.exp(torch_module.special.gammaln(1 - reflection_values))
        )
        result = torch_module.where(negative_mask, reflected_branch, positive_branch)

        zero_mask = values == 0
        signed_zero_inf = torch_module.where(
            torch_module.signbit(values),
            torch_module.full_like(values, -torch_module.inf),
            torch_module.full_like(values, torch_module.inf),
        )
        result = torch_module.where(zero_mask, signed_zero_inf, result)

        negative_integer_mask = negative_mask & (values == torch_module.floor(values))
        result = torch_module.where(
            negative_integer_mask,
            torch_module.full_like(values, torch_module.nan),
            result,
        )
        if out is not None:
            copy_ = getattr(out, "copy_", None)
            if copy_ is not None:
                copy_(result)
            else:
                out[...] = pytorch_backend.to_numpy(result)
            return out
        return result

    gamma.__name__ = getattr(original_gamma, "__name__", "gamma")
    gamma.__doc__ = getattr(original_gamma, "__doc__", None)
    gamma._pyrecest_arraylike_contract = True
    gamma._pyrecest_finite_gradient_contract = True
    pytorch_backend.gamma = gamma
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.gamma = gamma


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
    _patch_pytorch_gamma_autograd_contract(
        pytorch_backend,
        backend,
        torch_module,
    )


__all__ = ["patch_pytorch_one_hot_scalar_contract"]
