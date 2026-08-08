"""PyTorch ``one_hot``, ``take``, gamma, and equality compatibility hooks."""

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
        if labels.numel() == 0 and num_classes == 0:
            return torch_module.empty(
                (*labels.shape, 0),
                dtype=torch_module.uint8,
                device=labels.device,
            )
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
        reflection_sine = torch_module.sin(torch_module.pi * reflection_values)
        reflected_log_abs = (
            torch_module.log(
                torch_module.full_like(reflection_values, torch_module.pi)
            )
            - torch_module.log(torch_module.abs(reflection_sine))
            - torch_module.special.gammaln(1 - reflection_values)
        )
        reflected_branch = torch_module.sign(reflection_sine) * torch_module.exp(
            reflected_log_abs
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


def _patch_pytorch_array_equal_dtype_contract(
    pytorch_backend,
    backend,
    torch_module,
) -> None:
    """Make mixed-dtype ``array_equal`` follow NumPy promotion semantics."""
    original_array_equal = getattr(pytorch_backend, "array_equal", None)
    if original_array_equal is None:
        return
    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    if getattr(
        original_array_equal,
        "_pyrecest_numpy_dtype_promotion_contract",
        False,
    ):
        if active_pytorch_backend:
            backend.array_equal = original_array_equal
        return

    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - NumPy is a core dependency
        return

    torch_to_numpy_dtype = {}
    numpy_to_torch_dtype = {}
    for torch_name, numpy_dtype in (
        ("bool", np.bool_),
        ("uint8", np.uint8),
        ("uint16", np.uint16),
        ("uint32", np.uint32),
        ("uint64", np.uint64),
        ("int8", np.int8),
        ("int16", np.int16),
        ("int32", np.int32),
        ("int64", np.int64),
        ("float16", np.float16),
        ("float32", np.float32),
        ("float64", np.float64),
        ("complex64", np.complex64),
        ("complex128", np.complex128),
    ):
        torch_dtype = getattr(torch_module, torch_name, None)
        if torch_dtype is None:
            continue
        numpy_dtype = np.dtype(numpy_dtype)
        torch_to_numpy_dtype[torch_dtype] = numpy_dtype
        numpy_to_torch_dtype[numpy_dtype] = torch_dtype

    def _preferred_device(*values):
        for value in values:
            if torch_module.is_tensor(value) and value.device.type != "cpu":
                return value.device
        for value in values:
            if torch_module.is_tensor(value):
                return value.device
        return None

    def _coerce(value, *, device):
        if not torch_module.is_tensor(value):
            return torch_module.as_tensor(value, device=device)
        if device is not None and value.device != device:
            return value.to(device=device)
        return value

    def _promoted_dtype(first_dtype, second_dtype):
        first_numpy_dtype = torch_to_numpy_dtype.get(first_dtype)
        second_numpy_dtype = torch_to_numpy_dtype.get(second_dtype)
        if first_numpy_dtype is None or second_numpy_dtype is None:
            return torch_module.promote_types(first_dtype, second_dtype)
        promoted_numpy_dtype = np.dtype(
            np.result_type(first_numpy_dtype, second_numpy_dtype)
        )
        return numpy_to_torch_dtype.get(
            promoted_numpy_dtype,
            torch_module.promote_types(first_dtype, second_dtype),
        )

    def array_equal(a, b, equal_nan=False):
        device = _preferred_device(a, b)
        a = _coerce(a, device=device)
        b = _coerce(b, device=device)
        if tuple(a.shape) != tuple(b.shape):
            return False

        dtype = _promoted_dtype(a.dtype, b.dtype)
        a = a.to(dtype=dtype)
        b = b.to(dtype=dtype)
        if not equal_nan:
            return torch_module.equal(a, b)

        comparison = torch_module.eq(a, b)
        if dtype.is_floating_point or dtype.is_complex:
            comparison = comparison | (
                torch_module.isnan(a) & torch_module.isnan(b)
            )
        return bool(torch_module.all(comparison))

    array_equal.__name__ = getattr(original_array_equal, "__name__", "array_equal")
    array_equal.__doc__ = getattr(original_array_equal, "__doc__", None)
    array_equal._pyrecest_equal_nan_contract = True
    array_equal._pyrecest_numpy_contract = True
    array_equal._pyrecest_numpy_dtype_promotion_contract = True
    pytorch_backend.array_equal = array_equal
    if active_pytorch_backend:
        backend.array_equal = array_equal


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
    _patch_pytorch_array_equal_dtype_contract(
        pytorch_backend,
        backend,
        torch_module,
    )


__all__ = ["patch_pytorch_one_hot_scalar_contract"]
