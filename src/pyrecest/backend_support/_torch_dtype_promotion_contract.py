"""PyTorch backend compatibility helpers."""

from __future__ import annotations

from operator import index as _operator_index


def patch_pytorch_dtype_promotion_contract() -> None:
    """Make PyTorch backend helpers use PyRecEst compatibility contracts."""
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch.random as raw_pytorch_random  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
        from pyrecest._backend.pytorch._common import (  # pylint: disable=import-outside-toplevel
            _normalize_dtype,
        )
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - PyTorch backend import failed earlier
        return

    _patch_pytorch_repeat_numpy_contract(raw_pytorch, torch)
    _patch_pytorch_diff_numpy_contract(raw_pytorch, torch)
    _patch_pytorch_transpose_numpy_axes_contract(raw_pytorch, np)
    _patch_pytorch_pad_constant_values_contract(raw_pytorch, torch, np)
    _patch_pytorch_creation_numpy_contract(raw_pytorch, torch, np, _normalize_dtype)
    _patch_pytorch_randint_empty_size_contract(raw_pytorch_random, torch)

    original_convert = raw_pytorch.convert_to_wider_dtype
    if getattr(original_convert, "_pyrecest_torch_promotion_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.convert_to_wider_dtype = original_convert
        return

    def convert_to_wider_dtype(tensor_list):
        tensors = list(tensor_list)
        if not tensors:
            return tensors

        promoted_dtype = tensors[0].dtype
        for tensor in tensors[1:]:
            promoted_dtype = torch.promote_types(promoted_dtype, tensor.dtype)

        if all(tensor.dtype == promoted_dtype for tensor in tensors):
            return tensors
        return [raw_pytorch.cast(tensor, dtype=promoted_dtype) for tensor in tensors]

    convert_to_wider_dtype.__name__ = getattr(
        original_convert, "__name__", "convert_to_wider_dtype"
    )
    convert_to_wider_dtype.__doc__ = getattr(original_convert, "__doc__", None)
    convert_to_wider_dtype._pyrecest_torch_promotion_contract = True
    raw_pytorch.convert_to_wider_dtype = convert_to_wider_dtype
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.convert_to_wider_dtype = convert_to_wider_dtype


def _pytorch_repeat_count(repetition) -> int:
    """Return one NumPy-style repeat count as a non-negative integer."""
    try:
        count = _operator_index(repetition)
    except TypeError as exc:
        raise TypeError("repeat counts must be integers") from exc
    if count < 0:
        raise ValueError("repeats may not contain negative values")
    return count


def _pytorch_repeat_counts(repeats, *, numpy_module, torch_module, device):
    """Normalize NumPy-style repeat counts for ``torch.repeat_interleave``."""
    if torch_module.is_tensor(repeats):
        if repeats.ndim > 1:
            raise ValueError("object too deep for desired array")
        if repeats.dtype.is_floating_point or repeats.dtype.is_complex:
            raise TypeError("repeat counts must be integers")
        repeat_counts = repeats.to(device=device, dtype=torch_module.long)
        if bool(torch_module.any(repeat_counts < 0)):
            raise ValueError("repeats may not contain negative values")
        return repeat_counts

    repeats_array = numpy_module.asarray(repeats)
    if repeats_array.shape == ():
        return _pytorch_repeat_count(repeats_array.item())
    if repeats_array.ndim > 1:
        raise ValueError("object too deep for desired array")
    if not numpy_module.can_cast(
        repeats_array.dtype,
        numpy_module.dtype("intp"),
        casting="safe",
    ):
        raise TypeError("repeat counts must be integers")
    repeat_counts = torch_module.as_tensor(
        repeats_array,
        dtype=torch_module.long,
        device=device,
    )
    if bool(torch_module.any(repeat_counts < 0)):
        raise ValueError("repeats may not contain negative values")
    return repeat_counts


def _patch_pytorch_repeat_numpy_contract(raw_pytorch, torch) -> None:
    """Make raw/public PyTorch repeat follow the PyRecEst NumPy-style contract."""
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - NumPy is a core dependency
        return

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        backend = None

    original_repeat = raw_pytorch.repeat
    if getattr(original_repeat, "_pyrecest_numpy_contract", False):
        return

    def repeat(a, repeats, axis=None, *, dim=None, output_size=None):
        if dim is not None:
            if axis is not None and axis != dim:
                raise TypeError("repeat() got both 'axis' and 'dim'")
            axis = dim
        if axis is not None:
            axis = _operator_index(axis)

        values = raw_pytorch.array(a)
        repeat_counts = _pytorch_repeat_counts(
            repeats,
            numpy_module=np,
            torch_module=torch,
            device=values.device,
        )
        kwargs = {"dim": axis}
        if output_size is not None:
            kwargs["output_size"] = output_size
        return original_repeat(values, repeat_counts, **kwargs)

    repeat.__name__ = getattr(original_repeat, "__name__", "repeat")
    repeat.__doc__ = getattr(original_repeat, "__doc__", None)
    repeat._pyrecest_numpy_contract = True
    raw_pytorch.repeat = repeat
    if backend is not None and getattr(backend, "__backend_name__", None) == "pytorch":
        backend.repeat = repeat


def _patch_pytorch_diff_numpy_contract(raw_pytorch, torch) -> None:
    """Make raw/public PyTorch diff follow the PyRecEst NumPy-style contract."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        backend = None

    original_diff = raw_pytorch.diff
    if getattr(original_diff, "_pyrecest_numpy_contract", False):
        return
    no_boundary = object()

    def _normalize_axis(axis, ndim):
        axis = _operator_index(axis)
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise IndexError(
                f"axis {axis} is out of bounds for array of dimension {ndim}"
            )
        return axis

    def _boundary(value, reference, axis):
        boundary = raw_pytorch.array(value)
        if boundary.device != reference.device:
            boundary = boundary.to(device=reference.device)
        if boundary.ndim == 0:
            boundary_shape = list(reference.shape)
            boundary_shape[axis] = 1
            boundary = torch.broadcast_to(boundary, tuple(boundary_shape))
        return boundary

    def diff(a, n=1, axis=-1, prepend=no_boundary, append=no_boundary):
        values = raw_pytorch.array(a)
        order = _operator_index(n)
        if order < 0:
            raise ValueError(f"order must be non-negative but got {order}")
        if order == 0:
            return values.clone()
        if values.ndim == 0:
            raise ValueError("diff requires input that is at least one dimensional")

        axis = _normalize_axis(axis, values.ndim)
        diff_inputs = []
        if prepend is not no_boundary:
            diff_inputs.append(_boundary(prepend, values, axis))
        diff_inputs.append(values)
        if append is not no_boundary:
            diff_inputs.append(_boundary(append, values, axis))
        if len(diff_inputs) > 1:
            diff_inputs = raw_pytorch.convert_to_wider_dtype(diff_inputs)
            values = torch.cat(diff_inputs, dim=axis)
        return torch.diff(values, n=order, dim=axis)

    diff.__name__ = getattr(original_diff, "__name__", "diff")
    diff.__doc__ = getattr(original_diff, "__doc__", None)
    diff._pyrecest_numpy_contract = True
    raw_pytorch.diff = diff
    if backend is not None and getattr(backend, "__backend_name__", None) == "pytorch":
        backend.diff = diff


def _normalize_transpose_axes(axes, np):
    """Return PyTorch ``permute`` axes from NumPy-style transpose axes."""
    axes_array = np.asarray(axes)
    if axes_array.shape == () or axes_array.ndim != 1:
        raise ValueError("axes don't match array")
    return tuple(_operator_index(axis) for axis in axes_array.tolist())


def _patch_pytorch_transpose_numpy_axes_contract(raw_pytorch, np) -> None:
    """Make raw/public PyTorch transpose accept NumPy-style axes arrays."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        backend = None

    original_transpose = raw_pytorch.transpose
    if getattr(original_transpose, "_pyrecest_numpy_axes_contract", False):
        if (
            backend is not None
            and getattr(backend, "__backend_name__", None) == "pytorch"
        ):
            backend.transpose = original_transpose
        return

    def transpose(x, axes=None):
        values = raw_pytorch.array(x)
        if axes is None:
            return original_transpose(values, axes=None)
        return values.permute(_normalize_transpose_axes(axes, np))

    transpose.__name__ = getattr(original_transpose, "__name__", "transpose")
    transpose.__doc__ = getattr(original_transpose, "__doc__", None)
    transpose._pyrecest_numpy_axes_contract = True
    raw_pytorch.transpose = transpose
    if backend is not None and getattr(backend, "__backend_name__", None) == "pytorch":
        backend.transpose = transpose


def _normalize_creation_shape(shape, torch, np):
    """Return a tuple of Python integers for NumPy-style shape inputs."""
    if torch.is_tensor(shape):
        shape = shape.detach().cpu().numpy()
    shape_array = np.asarray(shape)
    if shape_array.shape == ():
        normalized_shape = (_operator_index(shape_array.item()),)
    else:
        normalized_shape = tuple(
            _operator_index(one_dimension) for one_dimension in shape_array.tolist()
        )
    if any(one_dimension < 0 for one_dimension in normalized_shape):
        raise ValueError("negative dimensions are not allowed")
    return normalized_shape


def _patch_pytorch_creation_numpy_contract(
    raw_pytorch,
    torch,
    np,
    normalize_dtype,
) -> None:
    """Make raw/public PyTorch creation helpers accept NumPy shapes and dtypes."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        backend = None

    active_pytorch_backend = (
        backend is not None and getattr(backend, "__backend_name__", None) == "pytorch"
    )

    def _wrap_creation_helper(helper_name, torch_helper, *, has_fill_value=False):
        original_helper = getattr(raw_pytorch, helper_name)
        if getattr(original_helper, "_pyrecest_numpy_contract", False):
            if active_pytorch_backend:
                setattr(backend, helper_name, original_helper)
            return original_helper

        if has_fill_value:

            def creation_helper(shape, fill_value, dtype=None, *args, **kwargs):
                return torch_helper(
                    _normalize_creation_shape(shape, torch, np),
                    fill_value,
                    *args,
                    dtype=normalize_dtype(dtype),
                    **kwargs,
                )

        else:

            def creation_helper(shape, dtype=None, *args, **kwargs):
                return torch_helper(
                    _normalize_creation_shape(shape, torch, np),
                    *args,
                    dtype=normalize_dtype(dtype),
                    **kwargs,
                )

        creation_helper.__name__ = getattr(original_helper, "__name__", helper_name)
        creation_helper.__doc__ = getattr(original_helper, "__doc__", None)
        creation_helper._pyrecest_numpy_contract = True
        return creation_helper

    for helper_name, torch_helper, has_fill_value in (
        ("empty", torch.empty, False),
        ("zeros", torch.zeros, False),
        ("ones", torch.ones, False),
        ("full", torch.full, True),
    ):
        helper = _wrap_creation_helper(
            helper_name,
            torch_helper,
            has_fill_value=has_fill_value,
        )
        setattr(raw_pytorch, helper_name, helper)
        if active_pytorch_backend:
            setattr(backend, helper_name, helper)


def _shape_has_zero_dimension(shape) -> bool:
    """Return whether a shape requests no samples."""
    return any(dimension == 0 for dimension in shape)


def _empty_randint_tensor(shape, *, dtype, device, out, torch):
    """Return or write an empty integer sample tensor."""
    result = torch.empty(shape, dtype=dtype, device=device)
    if out is not None:
        out.copy_(result)
        return out
    return result


def _empty_scalar_randint_result(shape, kwargs, raw_pytorch_random, torch):
    """Return an empty scalar-bound randint result using Torch keyword semantics."""
    kwargs = dict(kwargs)
    dtype = raw_pytorch_random._normalize_random_dtype(
        kwargs.pop("dtype", None),
        default=torch.int64,
    )
    device = kwargs.pop("device", None)
    generator = kwargs.pop("generator", None)
    out = kwargs.pop("out", None)
    layout = kwargs.pop("layout", torch.strided)
    requires_grad = kwargs.pop("requires_grad", False)
    pin_memory = kwargs.pop("pin_memory", False)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")
    del generator

    result = torch.empty(
        shape,
        dtype=dtype,
        device=device,
        layout=layout,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
    )
    if out is not None:
        out.copy_(result)
        return out
    return result


def _empty_array_randint_result_or_none(
    low,
    high,
    size,
    args,
    kwargs,
    raw_pytorch_random,
    torch,
):
    """Return an empty array-bound randint result, or ``None`` for non-empty draws."""
    if args:
        return None

    kwargs = dict(kwargs)
    dtype = raw_pytorch_random._normalize_random_dtype(
        kwargs.pop("dtype", None),
        default=torch.int64,
    )
    device = kwargs.pop("device", None)
    generator = kwargs.pop("generator", None)
    out = kwargs.pop("out", None)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")
    del generator

    device = raw_pytorch_random._randint_device(low, high, device=device)
    low = torch.as_tensor(low, device=device)
    high = torch.as_tensor(high, device=device)
    raw_pytorch_random._validate_randint_array_bound("low", low)
    raw_pytorch_random._validate_randint_array_bound("high", high)
    sample_shape = raw_pytorch_random._randint_array_size(size, low, high)
    try:
        torch.broadcast_to(low, sample_shape)
        torch.broadcast_to(high, sample_shape)
    except RuntimeError as exc:
        raise ValueError("size, low, and high could not be broadcast together") from exc

    if not _shape_has_zero_dimension(sample_shape):
        return None
    return _empty_randint_tensor(
        sample_shape,
        dtype=dtype,
        device=device,
        out=out,
        torch=torch,
    )


def _patch_pytorch_randint_empty_size_contract(raw_pytorch_random, torch) -> None:
    """Make PyTorch randint match NumPy for empty outputs with invalid bounds."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        backend = None

    active_pytorch_backend = (
        backend is not None and getattr(backend, "__backend_name__", None) == "pytorch"
    )
    backend_random = (
        getattr(backend, "random", None) if active_pytorch_backend else None
    )
    original_randint = raw_pytorch_random.randint
    if getattr(original_randint, "_pyrecest_empty_size_contract", False):
        if active_pytorch_backend and backend_random is not None:
            backend_random.randint = original_randint
        return

    def randint(low, high=None, size=None, *args, **kwargs):
        kwargs = raw_pytorch_random._normalize_torch_dtype_kwargs(kwargs)
        if high is None:
            if low is None:
                return original_randint(low, high, size, *args, **kwargs)
            if raw_pytorch_random._is_array_parameter(low):
                empty_result = _empty_array_randint_result_or_none(
                    0,
                    low,
                    size,
                    args,
                    kwargs,
                    raw_pytorch_random,
                    torch,
                )
                if empty_result is not None:
                    return empty_result
                return original_randint(low, high, size, *args, **kwargs)

            sample_shape = raw_pytorch_random._randint_size(size)
            if not args and _shape_has_zero_dimension(sample_shape):
                return _empty_scalar_randint_result(
                    sample_shape,
                    kwargs,
                    raw_pytorch_random,
                    torch,
                )
            return original_randint(low, high, size, *args, **kwargs)

        if raw_pytorch_random._is_array_parameter(
            low
        ) or raw_pytorch_random._is_array_parameter(high):
            empty_result = _empty_array_randint_result_or_none(
                low,
                high,
                size,
                args,
                kwargs,
                raw_pytorch_random,
                torch,
            )
            if empty_result is not None:
                return empty_result
            return original_randint(low, high, size, *args, **kwargs)

        sample_shape = raw_pytorch_random._randint_size(size)
        if not args and _shape_has_zero_dimension(sample_shape):
            return _empty_scalar_randint_result(
                sample_shape,
                kwargs,
                raw_pytorch_random,
                torch,
            )
        return original_randint(low, high, size, *args, **kwargs)

    randint.__name__ = getattr(original_randint, "__name__", "randint")
    randint.__doc__ = getattr(original_randint, "__doc__", None)
    randint._pyrecest_empty_size_contract = True
    raw_pytorch_random.randint = randint
    if active_pytorch_backend and backend_random is not None:
        backend_random.randint = randint


def _normalize_pad_pairs(pad_width, ndim, np):
    """Return NumPy-style per-axis pad-width pairs as Python integers."""
    pad_width_array = np.asarray(pad_width)
    if not np.issubdtype(pad_width_array.dtype, np.signedinteger):
        raise TypeError("pad_width must be of integral type")
    try:
        pad_pairs = np.broadcast_to(pad_width_array, (ndim, 2))
    except ValueError as exc:
        raise ValueError(
            f"pad_width must be broadcastable to shape ({ndim}, 2)"
        ) from exc
    if np.any(pad_pairs < 0):
        raise ValueError("index can't contain negative values")
    return tuple(
        (_operator_index(before), _operator_index(after))
        for before, after in pad_pairs.tolist()
    )


def _normalize_constant_value_pairs(constant_values, ndim, np):
    """Return NumPy-style per-axis constant-value pairs."""
    try:
        constant_pairs = np.broadcast_to(np.asarray(constant_values), (ndim, 2))
    except ValueError as exc:
        raise ValueError(
            f"constant_values must be broadcastable to shape ({ndim}, 2)"
        ) from exc
    return tuple(tuple(pair) for pair in constant_pairs.tolist())


def _filled_pad_block(shape, value, reference, torch):
    """Return a constant-filled block compatible with ``reference``."""
    scalar_value = torch.as_tensor(
        value, dtype=reference.dtype, device=reference.device
    )
    if scalar_value.ndim != 0:
        raise ValueError("constant_values entries must be scalar")
    block = torch.empty(tuple(shape), dtype=reference.dtype, device=reference.device)
    block.fill_(scalar_value)
    return block


def _constant_pad(values, pad_width, constant_values, torch, np):
    """Pad a tensor with NumPy-style per-axis constant values."""
    pad_pairs = _normalize_pad_pairs(pad_width, values.ndim, np)
    constant_pairs = _normalize_constant_value_pairs(constant_values, values.ndim, np)
    result = values
    for axis, ((before, after), (before_value, after_value)) in enumerate(
        zip(pad_pairs, constant_pairs)
    ):
        if before:
            before_shape = list(result.shape)
            before_shape[axis] = before
            before_block = _filled_pad_block(before_shape, before_value, result, torch)
            result = torch.cat((before_block, result), dim=axis)
        if after:
            after_shape = list(result.shape)
            after_shape[axis] = after
            after_block = _filled_pad_block(after_shape, after_value, result, torch)
            result = torch.cat((result, after_block), dim=axis)
    return result


def _patch_pytorch_pad_constant_values_contract(raw_pytorch, torch, np) -> None:
    """Make raw/public PyTorch pad accept NumPy-style constant_values."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        backend = None

    original_pad = raw_pytorch.pad
    if getattr(original_pad, "_pyrecest_constant_values_contract", False):
        return

    def pad(a, pad_width, mode="constant", constant_values=0.0):
        values = raw_pytorch.array(a)
        if mode != "constant":
            _normalize_pad_pairs(pad_width, values.ndim, np)
            return original_pad(
                values,
                pad_width,
                mode=mode,
                constant_values=constant_values,
            )
        return _constant_pad(values, pad_width, constant_values, torch, np)

    pad.__name__ = getattr(original_pad, "__name__", "pad")
    pad.__doc__ = getattr(original_pad, "__doc__", None)
    pad._pyrecest_constant_values_contract = True
    raw_pytorch.pad = pad
    if backend is not None and getattr(backend, "__backend_name__", None) == "pytorch":
        backend.pad = pad
