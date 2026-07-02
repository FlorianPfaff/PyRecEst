"""Utilities for exposing virtual backend submodules."""

from __future__ import annotations

import sys
from functools import wraps
from operator import index as _operator_index
from types import ModuleType

from pyrecest._backend import BACKEND_ATTRIBUTES


def _copy_result_to_out(result, out):
    """Copy ``result`` into a backend ``out`` object and return that object."""
    copy_ = getattr(out, "copy_", None)
    if copy_ is not None:
        copy_(result)
        return out
    try:
        out[...] = result
    except TypeError:
        at = getattr(out, "at", None)
        if at is None:
            raise
        return at[...].set(result)
    return out


def _cumulative_with_out(cumulative):
    """Return a cumulative helper accepting NumPy's ``out`` keyword."""

    @wraps(cumulative)
    def wrapped_cumulative(x, axis=None, dtype=None, out=None):
        result = cumulative(x, axis=axis, dtype=dtype)
        if out is not None:
            return _copy_result_to_out(result, out)
        return result

    wrapped_cumulative._pyrecest_out_contract = True
    return wrapped_cumulative


def _adapt_cumulative_out_contract(backend: ModuleType) -> None:
    """Adapt PyTorch cumulative helpers to the public NumPy-style contract."""
    if getattr(backend, "__backend_name__", None) != "pytorch":
        return
    for attribute_name in ("cumsum", "cumprod"):
        cumulative = getattr(backend, attribute_name, None)
        if cumulative is None or getattr(cumulative, "_pyrecest_out_contract", False):
            continue
        setattr(backend, attribute_name, _cumulative_with_out(cumulative))


def _adapt_pytorch_allclose_keyword_contract(backend: ModuleType) -> None:
    """Adapt raw and public PyTorch allclose to accept NumPy's missing-value keyword."""
    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch may be unavailable
        return

    allclose = getattr(pytorch_backend, "allclose", None)
    if allclose is None:
        return
    if getattr(allclose, "_pyrecest_missing_value_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            setattr(backend, "allclose", allclose)
        return

    missing_value_key = "_".join(("equal", "nan"))

    @wraps(allclose)
    def wrapped_allclose(
        a, b, atol=pytorch_backend.atol, rtol=pytorch_backend.rtol, **kwargs
    ):
        match_missing_values = kwargs.pop(missing_value_key, False)
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(
                f"allclose() got an unexpected keyword argument {unexpected!r}"
            )
        if not _torch.is_tensor(a):
            a = _torch.tensor(a)
        if not _torch.is_tensor(b):
            b = _torch.tensor(b)
        a, b = pytorch_backend.convert_to_wider_dtype([a, b])
        a, b = _torch.broadcast_tensors(a, b)
        return _torch.allclose(
            a,
            b,
            atol=atol,
            rtol=rtol,
            **{missing_value_key: match_missing_values},
        )

    wrapped_allclose._pyrecest_missing_value_contract = True
    setattr(pytorch_backend, "allclose", wrapped_allclose)
    if getattr(backend, "__backend_name__", None) == "pytorch":
        setattr(backend, "allclose", wrapped_allclose)


def _adapt_pytorch_isclose_keyword_contract(backend: ModuleType) -> None:
    """Adapt raw and public PyTorch isclose to accept NumPy's missing-value keyword."""
    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch may be unavailable
        return

    isclose = getattr(pytorch_backend, "isclose", None)
    if isclose is None:
        return
    if getattr(isclose, "_pyrecest_missing_value_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            setattr(backend, "isclose", isclose)
        return

    missing_value_key = "_".join(("equal", "nan"))

    @wraps(isclose)
    def wrapped_isclose(
        x, y, rtol=pytorch_backend.rtol, atol=pytorch_backend.atol, **kwargs
    ):
        match_missing_values = kwargs.pop(missing_value_key, False)
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(
                f"isclose() got an unexpected keyword argument {unexpected!r}"
            )
        if not _torch.is_tensor(x):
            x = _torch.tensor(x)
        if not _torch.is_tensor(y):
            y = _torch.tensor(y)
        x, y = pytorch_backend.convert_to_wider_dtype([x, y])
        x, y = _torch.broadcast_tensors(x, y)
        return _torch.isclose(
            x,
            y,
            atol=atol,
            rtol=rtol,
            **{missing_value_key: match_missing_values},
        )

    wrapped_isclose._pyrecest_missing_value_contract = True
    setattr(pytorch_backend, "isclose", wrapped_isclose)
    if getattr(backend, "__backend_name__", None) == "pytorch":
        setattr(backend, "isclose", wrapped_isclose)


def _coerce_pytorch_binary_tensor_args(pytorch_backend, torch_module, x, y):
    """Return PyTorch binary operands on a common device and dtype."""
    device = next(
        (value.device for value in (x, y) if torch_module.is_tensor(value)),
        None,
    )
    if not torch_module.is_tensor(x):
        x = torch_module.as_tensor(x, device=device)
    elif device is not None and x.device != device:
        x = x.to(device=device)
    if not torch_module.is_tensor(y):
        y = torch_module.as_tensor(y, device=device)
    elif device is not None and y.device != device:
        y = y.to(device=device)
    x, y = pytorch_backend.convert_to_wider_dtype([x, y])
    return x, y


def _adapt_pytorch_minmax_binary_contract(backend: ModuleType) -> None:
    """Adapt raw and public PyTorch maximum/minimum to array-like inputs."""
    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import torch as torch_module  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch may be unavailable
        return

    def _wrap_binary_minmax(torch_func, name):
        def wrapped_minmax(x, y):
            x, y = _coerce_pytorch_binary_tensor_args(
                pytorch_backend,
                torch_module,
                x,
                y,
            )
            return torch_func(x, y)

        wrapped_minmax.__name__ = name
        wrapped_minmax.__doc__ = getattr(torch_func, "__doc__", None)
        wrapped_minmax._pyrecest_binary_minmax_contract = True
        return wrapped_minmax

    for helper_name, torch_func in {
        "maximum": torch_module.maximum,
        "minimum": torch_module.minimum,
    }.items():
        current = getattr(pytorch_backend, helper_name, None)
        if current is None:
            continue
        if not getattr(current, "_pyrecest_binary_minmax_contract", False):
            current = _wrap_binary_minmax(torch_func, helper_name)
            setattr(pytorch_backend, helper_name, current)
        if getattr(backend, "__backend_name__", None) == "pytorch":
            setattr(backend, helper_name, current)


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
        repeats_array.dtype, numpy_module.dtype("intp"), casting="safe"
    ):
        raise TypeError("repeat counts must be integers")
    repeat_counts = torch_module.as_tensor(
        repeats_array, dtype=torch_module.long, device=device
    )
    if bool(torch_module.any(repeat_counts < 0)):
        raise ValueError("repeats may not contain negative values")
    return repeat_counts


def _pytorch_repeat_with_arraylike_inputs(
    repeat_interleave, array_func, numpy_module, torch_module
):
    """Return a NumPy-compatible ``repeat`` wrapper for the PyTorch backend."""

    @wraps(repeat_interleave)
    def repeat(a, repeats, axis=None, *, dim=None, output_size=None):
        if dim is not None:
            if axis is not None and axis != dim:
                raise TypeError("repeat() got both 'axis' and 'dim'")
            axis = dim
        if axis is not None:
            axis = _operator_index(axis)

        a = array_func(a)
        repeat_counts = _pytorch_repeat_counts(
            repeats,
            numpy_module=numpy_module,
            torch_module=torch_module,
            device=a.device,
        )
        kwargs = {"dim": axis}
        if output_size is not None:
            kwargs["output_size"] = output_size
        return repeat_interleave(a, repeat_counts, **kwargs)

    repeat._pyrecest_repeat_contract = True
    return repeat


def _adapt_pytorch_repeat_contract(backend: ModuleType) -> None:
    """Adapt PyTorch ``repeat`` to PyRecEst's NumPy-style backend contract."""
    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    import numpy as numpy_module  # pylint: disable=import-outside-toplevel
    import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
    import torch as torch_module  # pylint: disable=import-outside-toplevel

    repeat = getattr(pytorch_backend, "repeat", None)
    if repeat is None or getattr(repeat, "_pyrecest_repeat_contract", False):
        return
    wrapped_repeat = _pytorch_repeat_with_arraylike_inputs(
        repeat,
        backend.array,
        numpy_module,
        torch_module,
    )
    setattr(pytorch_backend, "repeat", wrapped_repeat)
    setattr(backend, "repeat", wrapped_repeat)


def _pytorch_reshape_shape(shape, torch_module) -> tuple[int, ...]:
    """Normalize NumPy-style reshape dimensions for ``torch.reshape``."""
    if torch_module.is_tensor(shape):
        if shape.ndim == 0:
            return (_operator_index(shape.item()),)
        shape = shape.detach().cpu().tolist()
    elif getattr(shape, "ndim", None) == 0 and hasattr(shape, "item"):
        return (_operator_index(shape.item()),)

    try:
        return (_operator_index(shape),)
    except TypeError:
        pass

    if isinstance(shape, (str, bytes)):
        raise TypeError("reshape shape must be an integer or a sequence of integers")

    try:
        return tuple(_operator_index(dimension) for dimension in shape)
    except TypeError as exc:
        raise TypeError(
            "reshape shape must be an integer or a sequence of integers"
        ) from exc


def _adapt_pytorch_reshape_contract(backend: ModuleType) -> None:
    """Adapt PyTorch reshape to accept array-like inputs and NumPy-style shapes."""
    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
    import torch as torch_module  # pylint: disable=import-outside-toplevel

    reshape = getattr(pytorch_backend, "reshape", None)
    if reshape is None or getattr(reshape, "_pyrecest_reshape_contract", False):
        return

    @wraps(reshape)
    def wrapped_reshape(x, shape):
        return reshape(backend.array(x), _pytorch_reshape_shape(shape, torch_module))

    wrapped_reshape._pyrecest_reshape_contract = True
    setattr(pytorch_backend, "reshape", wrapped_reshape)
    setattr(backend, "reshape", wrapped_reshape)


def _adapt_pytorch_stack_helpers_contract(backend: ModuleType) -> None:
    """Adapt raw PyTorch stack helpers to accept NumPy-style sequences."""
    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    try:
        import numpy as numpy_module  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import torch as torch_module  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - backend import fails first
        return

    helper_names = ("hstack", "vstack", "column_stack", "dstack")
    if all(
        getattr(
            getattr(pytorch_backend, helper_name, None),
            "_pyrecest_stack_contract",
            False,
        )
        for helper_name in helper_names
    ):
        return

    def _tensor_sequence(tup):
        return [backend.array(item) for item in tup]

    def hstack(tup):
        tensors = [torch_module.atleast_1d(tensor) for tensor in _tensor_sequence(tup)]
        if not tensors:
            return torch_module.cat(tensors, dim=0)
        return torch_module.cat(tensors, dim=0 if tensors[0].ndim == 1 else 1)

    def vstack(tup):
        tensors = [torch_module.atleast_2d(tensor) for tensor in _tensor_sequence(tup)]
        return torch_module.cat(tensors, dim=0)

    def column_stack(tup):
        tensors = []
        for tensor in _tensor_sequence(tup):
            if tensor.ndim < 2:
                tensor = tensor.reshape(-1, 1)
            tensors.append(tensor)
        return torch_module.cat(tensors, dim=1)

    def dstack(tup):
        tensors = [torch_module.atleast_3d(tensor) for tensor in _tensor_sequence(tup)]
        return torch_module.cat(tensors, dim=2)

    for helper_name, helper in {
        "hstack": hstack,
        "vstack": vstack,
        "column_stack": column_stack,
        "dstack": dstack,
    }.items():
        helper.__name__ = helper_name
        helper.__doc__ = getattr(numpy_module, helper_name).__doc__
        helper._pyrecest_stack_contract = True
        setattr(pytorch_backend, helper_name, helper)
        setattr(backend, helper_name, helper)


def _resolve_pytorch_reduction_axis(axis, dim, func_name):
    """Resolve NumPy ``axis`` and PyTorch ``dim`` aliases."""
    if dim is not None:
        if axis is not None and axis != dim:
            raise TypeError(f"{func_name}() got both 'axis' and 'dim'")
        axis = dim
    return axis


def _resolve_pytorch_keepdims(keepdims, keepdim, func_name):
    """Resolve NumPy ``keepdims`` and PyTorch ``keepdim`` aliases."""
    if keepdim is not None:
        if keepdims not in (False, None) and keepdims != keepdim:
            raise TypeError(f"{func_name}() got both 'keepdims' and 'keepdim'")
        keepdims = keepdim
    return keepdims


def _wrap_pytorch_axis_keepdim_reduction(reduction, func_name):
    """Return a reduction wrapper that accepts ``dim`` and ``keepdim``."""

    @wraps(reduction)
    def wrapped_reduction(
        x, axis=None, out=None, keepdims=False, *, dim=None, keepdim=None
    ):
        axis = _resolve_pytorch_reduction_axis(axis, dim, func_name)
        keepdims = _resolve_pytorch_keepdims(keepdims, keepdim, func_name)
        return reduction(x, axis=axis, out=out, keepdims=keepdims)

    wrapped_reduction._pyrecest_reduction_alias_contract = True
    return wrapped_reduction


def _wrap_pytorch_prod_reduction(prod):
    """Return a ``prod`` wrapper that accepts ``dim`` and ``keepdim``."""

    @wraps(prod)
    def wrapped_prod(
        x,
        axis=None,
        dtype=None,
        out=None,
        keepdims=False,
        *,
        dim=None,
        keepdim=None,
    ):
        axis = _resolve_pytorch_reduction_axis(axis, dim, "prod")
        keepdims = _resolve_pytorch_keepdims(keepdims, keepdim, "prod")
        return prod(x, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    wrapped_prod._pyrecest_reduction_alias_contract = True
    return wrapped_prod


def _wrap_pytorch_count_nonzero_reduction(count_nonzero):
    """Return a ``count_nonzero`` wrapper accepting ``dim`` and ``keepdim``."""

    @wraps(count_nonzero)
    def wrapped_count_nonzero(a, axis=None, keepdims=False, *, dim=None, keepdim=None):
        axis = _resolve_pytorch_reduction_axis(axis, dim, "count_nonzero")
        keepdims = _resolve_pytorch_keepdims(
            keepdims,
            keepdim,
            "count_nonzero",
        )
        return count_nonzero(a, axis=axis, keepdims=keepdims)

    wrapped_count_nonzero._pyrecest_reduction_alias_contract = True
    return wrapped_count_nonzero


def _adapt_pytorch_reduction_alias_contract(backend: ModuleType) -> None:
    """Preserve PyTorch ``dim``/``keepdim`` aliases on reduction helpers."""
    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel

    for reduction_name in ("all", "any", "max", "min"):
        reduction = getattr(pytorch_backend, reduction_name, None)
        if reduction is None or getattr(
            reduction,
            "_pyrecest_reduction_alias_contract",
            False,
        ):
            continue
        wrapped_reduction = _wrap_pytorch_axis_keepdim_reduction(
            reduction,
            reduction_name,
        )
        setattr(pytorch_backend, reduction_name, wrapped_reduction)
        setattr(backend, reduction_name, wrapped_reduction)

    prod = getattr(pytorch_backend, "prod", None)
    if prod is not None and not getattr(
        prod,
        "_pyrecest_reduction_alias_contract",
        False,
    ):
        wrapped_prod = _wrap_pytorch_prod_reduction(prod)
        setattr(pytorch_backend, "prod", wrapped_prod)
        setattr(backend, "prod", wrapped_prod)

    count_nonzero = getattr(pytorch_backend, "count_nonzero", None)
    if count_nonzero is not None and not getattr(
        count_nonzero,
        "_pyrecest_reduction_alias_contract",
        False,
    ):
        wrapped_count_nonzero = _wrap_pytorch_count_nonzero_reduction(count_nonzero)
        setattr(pytorch_backend, "count_nonzero", wrapped_count_nonzero)
        setattr(backend, "count_nonzero", wrapped_count_nonzero)

    if hasattr(pytorch_backend, "max"):
        setattr(pytorch_backend, "amax", pytorch_backend.max)
        setattr(backend, "amax", pytorch_backend.max)
    if hasattr(pytorch_backend, "min"):
        setattr(pytorch_backend, "amin", pytorch_backend.min)
        setattr(backend, "amin", pytorch_backend.min)


def _adapt_pytorch_cross_contract(backend: ModuleType) -> None:
    """Adapt PyTorch cross to match NumPy's 2D-vector result contract."""
    try:
        import numpy as numpy_module  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import torch as torch_module  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch may be unavailable
        return

    original_cross = getattr(pytorch_backend, "cross", None)
    if original_cross is None:
        return
    if getattr(original_cross, "_pyrecest_cross_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            setattr(backend, "cross", original_cross)
        return

    def _normalize_cross_axis(axis, ndim, name):
        axis = _operator_index(axis)
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise IndexError(
                f"{name} {axis} is out of bounds for array of dimension {ndim}"
            )
        return axis

    def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
        if axis is not None:
            axisa = axis
            axisb = axis
            axisc = axis

        a = pytorch_backend.array(a)
        b = pytorch_backend.array(b)
        a, b = pytorch_backend.convert_to_wider_dtype([a, b])

        axisa = _normalize_cross_axis(axisa, a.ndim, "axisa")
        axisb = _normalize_cross_axis(axisb, b.ndim, "axisb")
        a = torch_module.movedim(a, axisa, -1)
        b = torch_module.movedim(b, axisb, -1)

        a_dim = a.shape[-1]
        b_dim = b.shape[-1]
        if a_dim not in (2, 3) or b_dim not in (2, 3):
            raise ValueError(
                "incompatible dimensions for cross product "
                "(dimension must be 2 or 3)"
            )

        leading_shape = numpy_module.broadcast_shapes(
            tuple(a.shape[:-1]), tuple(b.shape[:-1])
        )
        if tuple(a.shape[:-1]) != leading_shape:
            a = torch_module.broadcast_to(a, leading_shape + (a_dim,))
        if tuple(b.shape[:-1]) != leading_shape:
            b = torch_module.broadcast_to(b, leading_shape + (b_dim,))

        z_component = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        if a_dim == 2 and b_dim == 2:
            return z_component

        if a_dim == 3 and b_dim == 3:
            result = torch_module.cross(a, b, dim=-1)
        elif a_dim == 2:
            result = pytorch_backend.stack(
                [
                    a[..., 1] * b[..., 2],
                    -a[..., 0] * b[..., 2],
                    z_component,
                ],
                dim=-1,
            )
        else:
            result = pytorch_backend.stack(
                [
                    -a[..., 2] * b[..., 1],
                    a[..., 2] * b[..., 0],
                    z_component,
                ],
                dim=-1,
            )

        axisc = _normalize_cross_axis(axisc, result.ndim, "axisc")
        return torch_module.movedim(result, -1, axisc)

    cross.__name__ = getattr(original_cross, "__name__", "cross")
    cross.__doc__ = getattr(original_cross, "__doc__", None)
    cross._pyrecest_cross_contract = True
    setattr(pytorch_backend, "cross", cross)
    if getattr(backend, "__backend_name__", None) == "pytorch":
        setattr(backend, "cross", cross)


def register_backend_submodules(backend: ModuleType | None = None) -> None:
    """Register virtual backend submodules for standard import statements."""
    if backend is None:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    _adapt_cumulative_out_contract(backend)
    _adapt_pytorch_allclose_keyword_contract(backend)
    _adapt_pytorch_isclose_keyword_contract(backend)
    _adapt_pytorch_minmax_binary_contract(backend)
    _adapt_pytorch_repeat_contract(backend)
    _adapt_pytorch_reshape_contract(backend)
    _adapt_pytorch_stack_helpers_contract(backend)
    _adapt_pytorch_reduction_alias_contract(backend)
    _adapt_pytorch_cross_contract(backend)

    backend.__path__ = getattr(backend, "__path__", [])
    backend_spec = getattr(backend, "__spec__", None)
    if backend_spec is not None:
        backend_spec.submodule_search_locations = (
            getattr(backend_spec, "submodule_search_locations", None) or []
        )

    for submodule_name in BACKEND_ATTRIBUTES:
        if not submodule_name:
            continue
        sys.modules[f"{backend.__name__}.{submodule_name}"] = getattr(
            backend, submodule_name
        )
