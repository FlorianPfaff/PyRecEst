"""Runtime backend contract patches that must run after backend support setup."""

from __future__ import annotations

from operator import index as _operator_index


def patch_pytorch_close_equal_nan_device_contract() -> None:
    """Preserve ``equal_nan`` while keeping PyTorch close operands on one device."""

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
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


def patch_pytorch_searchsorted_contract() -> None:
    """Restore PyTorch ``searchsorted`` with NumPy-style array-like inputs."""

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    original_searchsorted = getattr(raw_pytorch, "searchsorted", None)
    if getattr(original_searchsorted, "_pyrecest_searchsorted_contract", False):
        if active_pytorch_backend:
            backend.searchsorted = original_searchsorted
        return

    def _preferred_device(*values):
        for value in values:
            if torch.is_tensor(value) and value.device.type != "cpu":
                return value.device
        for value in values:
            if torch.is_tensor(value):
                return value.device
        return None

    def _as_ordered_tensor(value, *, device):
        if torch.is_tensor(value):
            return value.to(device=device) if device is not None else value
        return torch.as_tensor(value, device=device)

    def _as_sorter(sorter, *, device):
        if sorter is None:
            return None
        if torch.is_tensor(sorter):
            return sorter.to(device=device, dtype=torch.long)
        return torch.as_tensor(sorter, device=device, dtype=torch.long)

    def searchsorted(
        a,
        v,
        side="left",
        sorter=None,
        *,
        out=None,
        right=False,
        out_int32=False,
    ):
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        if right and side == "left":
            side = "right"

        device = _preferred_device(a, v, sorter)
        a = _as_ordered_tensor(a, device=device)
        v = _as_ordered_tensor(v, device=device)
        dtype = torch.promote_types(a.dtype, v.dtype)
        a = a.to(dtype=dtype)
        v = v.to(dtype=dtype)
        return torch.searchsorted(
            a,
            v,
            out_int32=out_int32,
            right=side == "right",
            out=out,
            sorter=_as_sorter(sorter, device=a.device),
        )

    searchsorted.__name__ = getattr(original_searchsorted, "__name__", "searchsorted")
    searchsorted.__doc__ = getattr(original_searchsorted, "__doc__", None)
    searchsorted._pyrecest_searchsorted_contract = True
    raw_pytorch.searchsorted = searchsorted
    if active_pytorch_backend:
        backend.searchsorted = searchsorted


def patch_pytorch_repeat_numpy_contract() -> None:
    """Preserve the raw PyTorch repeat contract for non-PyTorch public backends."""

    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
        from pyrecest._backend_submodules import (  # pylint: disable=import-outside-toplevel
            _pytorch_repeat_with_arraylike_inputs,
        )
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    _patch_pytorch_triangular_vector_rectangular_contract(raw_pytorch, backend)
    patch_pytorch_assignment_uint8_index_contract()

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
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
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
            raise ValueError(
                f"pad_width must be broadcastable to shape ({ndim}, 2)"
            ) from exc
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


def _normalize_sparse_target_shape(target_shape, np, torch) -> tuple[int, ...]:
    """Return a plain tuple shape for dense sparse-reconstruction output."""
    if torch.is_tensor(target_shape):
        target_shape = target_shape.detach().cpu().numpy()
    shape_array = np.asarray(target_shape)
    if shape_array.shape == ():
        normalized_shape = (_operator_index(shape_array.item()),)
    else:
        normalized_shape = tuple(_operator_index(size) for size in shape_array.tolist())
    if any(size < 0 for size in normalized_shape):
        raise ValueError("negative dimensions are not allowed")
    return normalized_shape


def _torch_sparse_indices(indices, target_shape, data, torch):
    """Return sparse coordinate indices with shape ``(n_entries, ndim)``."""
    indices = torch.as_tensor(indices, dtype=torch.long, device=data.device)
    if indices.numel() == 0:
        return indices
    if indices.ndim == 1 and len(target_shape) == 1:
        return indices.reshape(-1, 1)
    if indices.ndim != 2 or indices.shape[1] != len(target_shape):
        raise ValueError("indices must have shape (n_entries, ndim)")
    return indices


def _ravel_sparse_indices(indices, target_shape, torch):
    """Return C-order flat indices with NumPy ``ravel_multi_index`` checks."""
    shape_tensor = torch.as_tensor(
        target_shape,
        dtype=torch.long,
        device=indices.device,
    )
    if bool(torch.any(indices < 0)) or bool(torch.any(indices >= shape_tensor)):
        raise ValueError("invalid entry in coordinates array")

    strides = []
    stride = 1
    for size in reversed(target_shape):
        strides.insert(0, stride)
        stride *= size
    stride_tensor = torch.as_tensor(strides, dtype=torch.long, device=indices.device)
    return torch.sum(indices * stride_tensor, dim=1)


def patch_pytorch_array_from_sparse_assignment_contract() -> None:
    """Make PyTorch ``array_from_sparse`` match NumPy duplicate-index semantics."""

    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    original_array_from_sparse = getattr(raw_pytorch, "array_from_sparse", None)
    if original_array_from_sparse is None:
        return
    if getattr(
        original_array_from_sparse,
        "_pyrecest_sparse_assignment_contract",
        False,
    ):
        if active_pytorch_backend:
            backend.array_from_sparse = original_array_from_sparse
        return

    def array_from_sparse(indices, data, target_shape):
        data = raw_pytorch.array(data)
        target_shape = _normalize_sparse_target_shape(target_shape, np, torch)
        output = torch.zeros(
            torch.Size(target_shape),
            dtype=data.dtype,
            device=data.device,
        )
        sparse_indices = _torch_sparse_indices(indices, target_shape, data, torch)

        if sparse_indices.numel() == 0:
            if data.numel() != 0:
                raise ValueError("data must be empty when indices are empty")
            return output

        flat_indices = _ravel_sparse_indices(sparse_indices, target_shape, torch)
        output.reshape(-1)[flat_indices] = data
        return output

    array_from_sparse.__name__ = getattr(
        original_array_from_sparse,
        "__name__",
        "array_from_sparse",
    )
    array_from_sparse.__doc__ = getattr(original_array_from_sparse, "__doc__", None)
    array_from_sparse._pyrecest_sparse_assignment_contract = True
    raw_pytorch.array_from_sparse = array_from_sparse
    if active_pytorch_backend:
        backend.array_from_sparse = array_from_sparse


def patch_raw_pytorch_reduction_alias_contract() -> None:
    """Expose PyTorch ``dim``/``keepdim`` aliases on raw reductions always."""

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        from pyrecest._backend_submodules import (  # pylint: disable=import-outside-toplevel
            _wrap_pytorch_axis_keepdim_reduction,
            _wrap_pytorch_count_nonzero_reduction,
            _wrap_pytorch_prod_reduction,
        )
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    for reduction_name in ("all", "any", "max", "min"):
        reduction = getattr(raw_pytorch, reduction_name, None)
        if reduction is None:
            continue
        if not getattr(reduction, "_pyrecest_reduction_alias_contract", False):
            reduction = _wrap_pytorch_axis_keepdim_reduction(reduction, reduction_name)
            setattr(raw_pytorch, reduction_name, reduction)
        if active_pytorch_backend:
            setattr(backend, reduction_name, reduction)

    prod = getattr(raw_pytorch, "prod", None)
    if prod is not None:
        if not getattr(prod, "_pyrecest_reduction_alias_contract", False):
            prod = _wrap_pytorch_prod_reduction(prod)
            setattr(raw_pytorch, "prod", prod)
        if active_pytorch_backend:
            setattr(backend, "prod", prod)

    count_nonzero = getattr(raw_pytorch, "count_nonzero", None)
    if count_nonzero is not None:
        if not getattr(count_nonzero, "_pyrecest_reduction_alias_contract", False):
            count_nonzero = _wrap_pytorch_count_nonzero_reduction(count_nonzero)
            setattr(raw_pytorch, "count_nonzero", count_nonzero)
        if active_pytorch_backend:
            setattr(backend, "count_nonzero", count_nonzero)

    if hasattr(raw_pytorch, "max"):
        raw_pytorch.amax = raw_pytorch.max
        if active_pytorch_backend:
            backend.amax = raw_pytorch.max
    if hasattr(raw_pytorch, "min"):
        raw_pytorch.amin = raw_pytorch.min
        if active_pytorch_backend:
            backend.amin = raw_pytorch.min


def patch_pytorch_transpose_boolean_axes_contract() -> None:
    """Make PyTorch ``transpose`` reject boolean axes sequences like NumPy."""

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
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


def patch_jax_take_arraylike_contract() -> None:
    """Make raw/public JAX ``take`` accept NumPy-style array-like inputs."""

    try:
        import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.jax as raw_jax  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - JAX backend may be unavailable
        return

    original_take = getattr(raw_jax, "take", None)
    if original_take is None:
        return
    if getattr(original_take, "_pyrecest_arraylike_contract", False):
        if getattr(backend, "__backend_name__", None) == "jax":
            backend.take = original_take
        return

    def take(
        a,
        indices,
        axis=None,
        out=None,
        mode=None,
        unique_indices=False,
        indices_are_sorted=False,
        fill_value=None,
    ):
        result = original_take(
            jnp.asarray(a),
            jnp.asarray(indices),
            axis=axis,
            out=None,
            mode=mode,
            unique_indices=unique_indices,
            indices_are_sorted=indices_are_sorted,
            fill_value=fill_value,
        )
        if out is not None:
            return jnp.asarray(out).at[...].set(result)
        return result

    take.__name__ = getattr(original_take, "__name__", "take")
    take.__doc__ = getattr(original_take, "__doc__", None)
    take._pyrecest_arraylike_contract = True
    take._pyrecest_out_contract = True
    raw_jax.take = take
    if getattr(backend, "__backend_name__", None) == "jax":
        backend.take = take


def patch_pytorch_assignment_uint8_index_contract() -> None:
    """Treat uint8 PyTorch assignment indices as integer indices, not masks."""

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    current_is_boolean = getattr(raw_pytorch, "_is_boolean", None)
    current_as_assignment_index = getattr(raw_pytorch, "_as_assignment_index", None)
    if getattr(
        current_is_boolean, "_pyrecest_uint8_assignment_index_contract", False
    ) and getattr(
        current_as_assignment_index,
        "_pyrecest_uint8_assignment_index_contract",
        False,
    ):
        return

    def _is_boolean(x):
        if isinstance(x, bool):
            return True
        if isinstance(x, (tuple, list)):
            if not x:
                return False
            return _is_boolean(x[0])
        if torch.is_tensor(x):
            return x.dtype == torch.bool
        return False

    def _as_assignment_index(index, *, device):
        if torch.is_tensor(index):
            if index.dtype == torch.bool:
                return index.to(device=device)
            return index.to(device=device, dtype=torch.long)
        return torch.as_tensor(index, dtype=torch.long, device=device)

    _is_boolean._pyrecest_uint8_assignment_index_contract = True
    _as_assignment_index._pyrecest_uint8_assignment_index_contract = True
    raw_pytorch._is_boolean = _is_boolean
    raw_pytorch._as_assignment_index = _as_assignment_index
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.assignment = raw_pytorch.assignment
        backend.assignment_by_sum = raw_pytorch.assignment_by_sum


def patch_pytorch_take_axis_contract() -> None:
    """Make PyTorch ``take`` reject non-integer axes like NumPy."""

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    original_take = getattr(raw_pytorch, "take", None)
    if original_take is None:
        return
    if getattr(original_take, "_pyrecest_axis_index_contract", False):
        if active_pytorch_backend:
            backend.take = original_take
        return

    def _take_axis(axis):
        if isinstance(axis, bool) or (
            torch.is_tensor(axis) and axis.ndim == 0 and axis.dtype == torch.bool
        ):
            raise TypeError("an integer is required for the axis")
        return _operator_index(axis)

    def take(a, indices, axis=None, out=None, mode=None):
        axis = None if axis is None else _take_axis(axis)
        return original_take(a, indices, axis=axis, out=out, mode=mode)

    take.__name__ = getattr(original_take, "__name__", "take")
    take.__doc__ = getattr(original_take, "__doc__", None)
    take._pyrecest_axis_index_contract = True
    raw_pytorch.take = take
    if active_pytorch_backend:
        backend.take = take


patch_jax_take_arraylike_contract()
patch_pytorch_array_from_sparse_assignment_contract()
patch_pytorch_assignment_uint8_index_contract()
patch_pytorch_take_axis_contract()
