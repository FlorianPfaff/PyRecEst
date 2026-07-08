"""Runtime backend contract patches that must run after backend support setup."""

from __future__ import annotations

from operator import index as _operator_index


def patch_pytorch_close_equal_nan_device_contract() -> None:
    """Preserve ``equal_nan`` while keeping PyTorch close operands on one device."""

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    helper_names = ("isclose", "allclose")
    if all(
        getattr(
            getattr(raw_pytorch, helper_name, None),
            "_pyrecest_close_equal_nan_device_contract",
            False,
        )
        for helper_name in helper_names
    ):
        if active_pytorch_backend:
            for helper_name in helper_names:
                setattr(backend, helper_name, getattr(raw_pytorch, helper_name))
        return

    def _preferred_device(*values):
        for value in values:
            if torch.is_tensor(value) and value.device.type != "cpu":
                return value.device
        for value in values:
            if torch.is_tensor(value):
                return value.device
        return None

    def _tensor_on_device(value, *, device):
        if torch.is_tensor(value):
            if device is not None and value.device != device:
                return value.to(device=device)
            return value
        return torch.as_tensor(value, device=device)

    def _comparison_operands(a, b):
        device = _preferred_device(a, b)
        a = _tensor_on_device(a, device=device)
        b = _tensor_on_device(b, device=device)
        dtype = torch.promote_types(a.dtype, b.dtype)
        a = a.to(dtype=dtype)
        b = b.to(dtype=dtype)
        return torch.broadcast_tensors(a, b)

    def isclose(a, b, rtol=raw_pytorch.rtol, atol=raw_pytorch.atol, equal_nan=False):
        a, b = _comparison_operands(a, b)
        return torch.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def allclose(a, b, atol=raw_pytorch.atol, rtol=raw_pytorch.rtol, equal_nan=False):
        a, b = _comparison_operands(a, b)
        return torch.allclose(a, b, atol=atol, rtol=rtol, equal_nan=equal_nan)

    for helper_name, helper in {
        "isclose": isclose,
        "allclose": allclose,
    }.items():
        previous = getattr(raw_pytorch, helper_name, None)
        helper.__name__ = getattr(previous, "__name__", helper_name)
        helper.__doc__ = getattr(previous, "__doc__", None)
        helper._pyrecest_device_contract = True
        helper._pyrecest_missing_value_contract = True
        helper._pyrecest_close_equal_nan_device_contract = True
        setattr(raw_pytorch, helper_name, helper)
        if active_pytorch_backend:
            setattr(backend, helper_name, helper)


def _patch_pytorch_triangular_vector_rectangular_contract(
    raw_pytorch,
    backend,
) -> None:
    """Make PyTorch triangular-vector helpers use both matrix dimensions."""

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"

    for helper_name, index_helper_name in (
        ("tril_to_vec", "tril_indices"),
        ("triu_to_vec", "triu_indices"),
    ):
        original_helper = getattr(raw_pytorch, helper_name, None)
        index_helper = getattr(raw_pytorch, index_helper_name, None)
        if original_helper is None or index_helper is None:
            continue
        if getattr(original_helper, "_pyrecest_rectangular_matrix_contract", False):
            if active_pytorch_backend:
                setattr(backend, helper_name, original_helper)
            continue

        def triangular_to_vec(x, k=0, *, _index=index_helper):
            values = raw_pytorch.array(x)
            if values.ndim < 2:
                raise ValueError(
                    "triangular vector helpers require at least two matrix dimensions"
                )
            rows, cols = _index(values.shape[-2], k=k, m=values.shape[-1])
            rows = rows.to(device=values.device)
            cols = cols.to(device=values.device)
            return values[..., rows, cols]

        triangular_to_vec.__name__ = getattr(original_helper, "__name__", helper_name)
        triangular_to_vec.__doc__ = getattr(original_helper, "__doc__", None)
        triangular_to_vec._pyrecest_arraylike_contract = True
        triangular_to_vec._pyrecest_rectangular_matrix_contract = True
        setattr(raw_pytorch, helper_name, triangular_to_vec)
        if active_pytorch_backend:
            setattr(backend, helper_name, triangular_to_vec)


def patch_pytorch_repeat_numpy_contract() -> None:
    """Preserve the raw PyTorch repeat contract for non-PyTorch public backends."""

    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
        from pyrecest._backend_submodules import (  # pylint: disable=import-outside-toplevel
            _pytorch_repeat_with_arraylike_inputs,
        )
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    _patch_pytorch_triangular_vector_rectangular_contract(raw_pytorch, backend)

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    original_repeat = getattr(raw_pytorch, "repeat", None)
    if original_repeat is None:
        return
    if getattr(original_repeat, "_pyrecest_repeat_contract", False):
        if active_pytorch_backend:
            setattr(backend, "repeat", original_repeat)
        return

    repeat = _pytorch_repeat_with_arraylike_inputs(
        original_repeat,
        raw_pytorch.array,
        np,
        torch,
    )
    raw_pytorch.repeat = repeat
    if active_pytorch_backend:
        backend.repeat = repeat


def patch_pytorch_edge_pad_contract() -> None:
    """Implement NumPy-style ``pad(..., mode='edge')`` for the PyTorch backend."""

    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    original_pad = getattr(raw_pytorch, "pad", None)
    if original_pad is None:
        return
    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    if getattr(original_pad, "_pyrecest_edge_mode_contract", False):
        if active_pytorch_backend:
            backend.pad = original_pad
        return

    def _normalize_pad_pairs(pad_width, ndim):
        pad_width_array = np.asarray(pad_width)
        if not np.issubdtype(pad_width_array.dtype, np.signedinteger):
            raise TypeError("pad_width must be of integral type")
        try:
            pad_pairs = np.broadcast_to(pad_width_array, (ndim, 2))
        except ValueError as exc:
            raise ValueError(f"pad_width must be broadcastable to shape ({ndim}, 2)") from exc
        if np.any(pad_pairs < 0):
            raise ValueError("index can't contain negative values")
        return tuple((int(before), int(after)) for before, after in pad_pairs.tolist())

    def _edge_block(values, axis, width, edge_index):
        edge = torch.narrow(values, axis, edge_index, 1)
        shape = list(values.shape)
        shape[axis] = width
        return torch.broadcast_to(edge, tuple(shape))

    def _edge_pad(values, pad_width):
        result = values
        for axis, (before, after) in enumerate(
            _normalize_pad_pairs(pad_width, values.ndim)
        ):
            if (before or after) and result.shape[axis] == 0:
                raise ValueError(
                    f"can't extend empty axis {axis} using modes other than 'constant' or 'empty'"
                )
            if before:
                result = torch.cat(
                    (_edge_block(result, axis, before, 0), result),
                    dim=axis,
                )
            if after:
                result = torch.cat(
                    (
                        result,
                        _edge_block(result, axis, after, result.shape[axis] - 1),
                    ),
                    dim=axis,
                )
        return result

    def pad(a, pad_width, mode="constant", constant_values=0.0):
        if mode != "edge":
            return original_pad(
                a,
                pad_width,
                mode=mode,
                constant_values=constant_values,
            )
        values = raw_pytorch.array(a)
        return _edge_pad(values, pad_width)

    pad.__name__ = getattr(original_pad, "__name__", "pad")
    pad.__doc__ = getattr(original_pad, "__doc__", None)
    pad._pyrecest_edge_mode_contract = True
    raw_pytorch.pad = pad
    if active_pytorch_backend:
        backend.pad = pad


def patch_pytorch_transpose_boolean_axes_contract() -> None:
    """Make PyTorch ``transpose`` reject boolean axes sequences like NumPy."""

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    original_transpose = getattr(raw_pytorch, "transpose", None)
    if original_transpose is None:
        return
    if getattr(original_transpose, "_pyrecest_transpose_boolean_axes_contract", False):
        if active_pytorch_backend:
            setattr(backend, "transpose", original_transpose)
        return

    def _normalized_axes(axes):
        if axes is None:
            return None
        if torch.is_tensor(axes):
            if axes.ndim == 0:
                axes = axes.item()
            else:
                axes = axes.detach().cpu().tolist()
        if isinstance(axes, (str, bytes)):
            raise TypeError("transpose axes must be a sequence of integers")
        try:
            iterator = iter(axes)
        except TypeError as exc:
            raise TypeError("transpose axes must be a sequence of integers") from exc

        normalized = []
        for axis in iterator:
            if isinstance(axis, bool) or (
                torch.is_tensor(axis) and axis.ndim == 0 and axis.dtype == torch.bool
            ):
                raise TypeError("an integer is required")
            normalized.append(_operator_index(axis))
        return tuple(normalized)

    def transpose(x, axes=None):
        return original_transpose(x, axes=_normalized_axes(axes))

    transpose.__name__ = getattr(original_transpose, "__name__", "transpose")
    transpose.__doc__ = getattr(original_transpose, "__doc__", None)
    transpose._pyrecest_transpose_boolean_axes_contract = True
    raw_pytorch.transpose = transpose
    if active_pytorch_backend:
        backend.transpose = transpose
