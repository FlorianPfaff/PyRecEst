"""PyTorch backend logical-helper compatibility patch."""

from __future__ import annotations

import importlib.util
from operator import index as _operator_index
from pathlib import Path


def _load_base_contract_module():
    module_path = Path(__file__).resolve().parent.parent / "_torch_dtype_promotion_contract.py"
    spec = importlib.util.spec_from_file_location(
        "_pyrecest_torch_dtype_promotion_contract_base",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load PyTorch dtype contract module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE_CONTRACT = _load_base_contract_module()


def patch_pytorch_dtype_promotion_contract() -> None:
    """Apply the base PyTorch contract patch plus device-placement fixes."""
    _BASE_CONTRACT.patch_pytorch_dtype_promotion_contract()
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    _patch_pytorch_assignment_numpy_index_contract(raw_pytorch, backend, torch, np)
    _patch_pytorch_logical_device_contract(raw_pytorch, backend, torch)
    _patch_pytorch_comparison_device_contract(raw_pytorch, backend, torch)
    _patch_pytorch_binary_device_contract(raw_pytorch, backend, torch)
    _patch_pytorch_equality_device_contract(raw_pytorch, backend, torch)
    _patch_pytorch_linspace_integer_dtype_contract(raw_pytorch, backend, torch)
    _patch_pytorch_creation_bool_shape_contract(raw_pytorch, backend, torch, np)
    _patch_pytorch_arraylike_helper_contract(raw_pytorch, backend, torch)
    _patch_pytorch_concatenate_axis_none_contract(raw_pytorch, backend, torch)


def _pytorch_numpy_index_array(index, numpy_module, torch_module):
    """Return tensor indices for NumPy index arrays before helper len() checks."""
    if not isinstance(index, numpy_module.ndarray):
        return index
    if numpy_module.issubdtype(index.dtype, numpy_module.bool_):
        return torch_module.as_tensor(index, dtype=torch_module.bool)
    if numpy_module.issubdtype(index.dtype, numpy_module.integer):
        return torch_module.as_tensor(index, dtype=torch_module.long)
    return index


def _is_pytorch_scalar_coordinate_index(index, torch_module, numpy_module):
    """Return whether ``index`` is an integer scalar coordinate component."""
    if isinstance(index, (bool, numpy_module.bool_)):
        return False

    if torch_module.is_tensor(index):
        if index.ndim != 0:
            return False
        if (
            index.dtype in {torch_module.bool, torch_module.uint8}
            or index.dtype.is_floating_point
            or index.dtype.is_complex
        ):
            return False
        return True

    if isinstance(index, numpy_module.ndarray):
        if index.shape != ():
            return False
        if (
            numpy_module.issubdtype(index.dtype, numpy_module.bool_)
            or numpy_module.issubdtype(index.dtype, numpy_module.floating)
            or numpy_module.issubdtype(index.dtype, numpy_module.complexfloating)
        ):
            return False
        index = index.item()

    try:
        _operator_index(index)
    except TypeError:
        return False
    return True


def _pytorch_scalar_coordinate_index(index, torch_module, numpy_module) -> int:
    """Return a Python int coordinate component."""
    if torch_module.is_tensor(index):
        index = index.item()
    elif isinstance(index, numpy_module.ndarray):
        index = index.item()
    return _operator_index(index)


def _pytorch_singleton_coordinate_sequence(indices, torch_module, numpy_module):
    """Normalize ``[(i, j, ...)]`` to ``(i, j, ...)`` before vectorization checks."""
    if not isinstance(indices, (list, tuple)) or len(indices) != 1:
        return None

    coordinate = indices[0]
    if not isinstance(coordinate, (list, tuple)) or not coordinate:
        return None

    if not all(
        _is_pytorch_scalar_coordinate_index(component, torch_module, numpy_module)
        for component in coordinate
    ):
        return None

    return tuple(
        _pytorch_scalar_coordinate_index(component, torch_module, numpy_module)
        for component in coordinate
    )


def _wrap_assignment_numpy_index_helper(original_helper, torch_module, numpy_module):
    """Normalize NumPy index arrays before assignment helper len() checks."""
    if getattr(original_helper, "_pyrecest_numpy_index_contract", False):
        return original_helper

    def assignment(x, values, indices, axis=0):
        singleton_coordinate = _pytorch_singleton_coordinate_sequence(
            indices,
            torch_module,
            numpy_module,
        )
        if singleton_coordinate is not None:
            indices = singleton_coordinate
        else:
            indices = _pytorch_numpy_index_array(indices, numpy_module, torch_module)
        return original_helper(x, values, indices, axis=axis)

    assignment.__name__ = getattr(original_helper, "__name__", "assignment")
    assignment.__doc__ = getattr(original_helper, "__doc__", None)
    assignment._pyrecest_numpy_index_contract = True
    return assignment


def _patch_pytorch_assignment_numpy_index_contract(raw_pytorch, backend, torch, np) -> None:
    """Make PyTorch assignment helpers accept NumPy integer and boolean indices."""
    helper_names = ("assignment", "assignment_by_sum")
    if all(
        getattr(getattr(raw_pytorch, helper_name, None), "_pyrecest_numpy_index_contract", False)
        for helper_name in helper_names
    ):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            for helper_name in helper_names:
                setattr(backend, helper_name, getattr(raw_pytorch, helper_name))
        return
    for helper_name in helper_names:
        wrapped_helper = _wrap_assignment_numpy_index_helper(
            getattr(raw_pytorch, helper_name),
            torch,
            np,
        )
        setattr(raw_pytorch, helper_name, wrapped_helper)
        if getattr(backend, "__backend_name__", None) == "pytorch":
            setattr(backend, helper_name, wrapped_helper)


def _preferred_pytorch_device(torch_module, *values):
    """Return a non-CPU tensor device when mixed-device operands are present."""
    for value in values:
        if torch_module.is_tensor(value) and value.device.type != "cpu":
            return value.device
    for value in values:
        if torch_module.is_tensor(value):
            return value.device
    return None


def _as_pytorch_tensor_on_device(value, torch_module, *, device, dtype=None):
    """Return ``value`` as a tensor on ``device``."""
    if torch_module.is_tensor(value):
        if device is not None and value.device != device:
            value = value.to(device=device)
        if dtype is not None and value.dtype != dtype:
            value = value.to(dtype=dtype)
        return value
    return torch_module.as_tensor(value, dtype=dtype, device=device)


def _patch_pytorch_logical_device_contract(raw_pytorch, backend, torch) -> None:
    """Keep logical helpers on an existing non-CPU tensor device."""
    helper_names = ("logical_and", "where")
    if all(
        getattr(
            getattr(raw_pytorch, helper_name, None),
            "_pyrecest_device_contract",
            False,
        )
        for helper_name in helper_names
    ):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            for helper_name in helper_names:
                setattr(backend, helper_name, getattr(raw_pytorch, helper_name))
        return
    original_logical_and = raw_pytorch.logical_and
    original_where = raw_pytorch.where

    def logical_and(x, y):
        device = _preferred_pytorch_device(torch, x, y)
        return torch.logical_and(
            _as_pytorch_tensor_on_device(x, torch, device=device),
            _as_pytorch_tensor_on_device(y, torch, device=device),
        )

    def where(condition, x=None, y=None):
        device = _preferred_pytorch_device(torch, condition, x, y)
        condition = _as_pytorch_tensor_on_device(
            condition,
            torch,
            device=device,
            dtype=torch.bool,
        )

        if x is None and y is None:
            return torch.where(condition)
        if x is None or y is None:
            raise ValueError("either both or neither of x and y should be given")

        x = _as_pytorch_tensor_on_device(x, torch, device=device)
        y = _as_pytorch_tensor_on_device(y, torch, device=device)
        result_dtype = torch.result_type(x, y)
        return torch.where(
            condition,
            x.to(dtype=result_dtype),
            y.to(dtype=result_dtype),
        )

    logical_and.__name__ = getattr(original_logical_and, "__name__", "logical_and")
    logical_and.__doc__ = getattr(original_logical_and, "__doc__", None)
    logical_and._pyrecest_device_contract = True
    where.__name__ = getattr(original_where, "__name__", "where")
    where.__doc__ = getattr(original_where, "__doc__", None)
    where._pyrecest_device_contract = True

    raw_pytorch.logical_and = logical_and
    raw_pytorch.where = where
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.logical_and = logical_and
        backend.where = where


def _is_array_like_operand(value):
    if isinstance(value, (str, bytes)):
        return False
    return isinstance(value, (list, tuple)) or hasattr(value, "__array__")


def _binary_operand(value, torch_module, *, box_array_like, device):
    if torch_module.is_tensor(value):
        if device is not None and value.device != device:
            return value.to(device=device)
        return value
    if box_array_like and _is_array_like_operand(value):
        return torch_module.as_tensor(value, device=device)
    return value


def _wrap_binary_device_helper(original_helper, torch_module, *, box_x2):
    def binary_helper(x1, x2, *args, **kwargs):
        device = _preferred_pytorch_device(torch_module, x1, x2)
        x1 = _binary_operand(x1, torch_module, box_array_like=True, device=device)
        x2 = _binary_operand(x2, torch_module, box_array_like=box_x2, device=device)
        return original_helper(x1, x2, *args, **kwargs)

    binary_helper.__name__ = getattr(original_helper, "__name__", "binary_helper")
    binary_helper.__doc__ = getattr(original_helper, "__doc__", None)
    binary_helper._pyrecest_device_contract = True
    return binary_helper


def _wrap_tensor_binary_device_helper(original_helper, torch_module):
    def binary_helper(x1, x2, *args, **kwargs):
        device = _preferred_pytorch_device(torch_module, x1, x2)
        x1 = _as_pytorch_tensor_on_device(x1, torch_module, device=device)
        x2 = _as_pytorch_tensor_on_device(x2, torch_module, device=device)
        return original_helper(x1, x2, *args, **kwargs)

    binary_helper.__name__ = getattr(original_helper, "__name__", "binary_helper")
    binary_helper.__doc__ = getattr(original_helper, "__doc__", None)
    binary_helper._pyrecest_device_contract = True
    return binary_helper


def _patch_pytorch_comparison_device_contract(raw_pytorch, backend, torch) -> None:
    """Make PyTorch comparison helpers accept NumPy-style array-like inputs."""
    helper_names = ("greater", "less", "logical_or")
    if all(
        getattr(getattr(raw_pytorch, helper_name, None), "_pyrecest_device_contract", False)
        for helper_name in helper_names
    ):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            for helper_name in helper_names:
                setattr(backend, helper_name, getattr(raw_pytorch, helper_name))
        return
    for helper_name in helper_names:
        wrapped_helper = _wrap_tensor_binary_device_helper(
            getattr(raw_pytorch, helper_name),
            torch,
        )
        setattr(raw_pytorch, helper_name, wrapped_helper)
        if getattr(backend, "__backend_name__", None) == "pytorch":
            setattr(backend, helper_name, wrapped_helper)


def _patch_pytorch_binary_device_contract(raw_pytorch, backend, torch) -> None:
    """Keep boxed PyTorch binary helper operands on an existing non-CPU device."""
    helpers = {
        "arctan2": True,
        "mod": False,
        "power": False,
    }
    if all(
        getattr(getattr(raw_pytorch, helper_name, None), "_pyrecest_device_contract", False)
        for helper_name in helpers
    ):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            for helper_name in helpers:
                setattr(backend, helper_name, getattr(raw_pytorch, helper_name))
        return

    for helper_name, box_x2 in helpers.items():
        wrapped_helper = _wrap_binary_device_helper(
            getattr(raw_pytorch, helper_name),
            torch,
            box_x2=box_x2,
        )
        setattr(raw_pytorch, helper_name, wrapped_helper)
        if getattr(backend, "__backend_name__", None) == "pytorch":
            setattr(backend, helper_name, wrapped_helper)


def _patch_pytorch_equality_device_contract(raw_pytorch, backend, torch) -> None:
    """Keep equality-style helpers on an existing non-CPU tensor device."""
    helper_names = ("equal", "less_equal", "array" + "_equal")
    if all(
        getattr(getattr(raw_pytorch, helper_name, None), "_pyrecest_device_contract", False)
        for helper_name in helper_names
    ):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            for helper_name in helper_names:
                setattr(backend, helper_name, getattr(raw_pytorch, helper_name))
        return

    for helper_name in helper_names:
        wrapped_helper = _wrap_tensor_binary_device_helper(
            getattr(raw_pytorch, helper_name),
            torch,
        )
        setattr(raw_pytorch, helper_name, wrapped_helper)
        if getattr(backend, "__backend_name__", None) == "pytorch":
            setattr(backend, helper_name, wrapped_helper)


def _integer_torch_dtype(dtype, raw_pytorch, torch):
    """Return an explicit integer torch dtype, or ``None`` for non-integers."""
    if dtype is None:
        return None
    try:
        torch_dtype = raw_pytorch.as_dtype(dtype)
    except (KeyError, TypeError, ValueError):
        return None
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    return torch_dtype if torch_dtype in integer_dtypes else None


def _patch_pytorch_linspace_integer_dtype_contract(raw_pytorch, backend, torch) -> None:
    """Make PyTorch linspace match NumPy flooring for explicit integer dtypes."""
    original_linspace = raw_pytorch.linspace
    if getattr(original_linspace, "_pyrecest_integer_dtype_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.linspace = original_linspace
        return
    def linspace(start, stop, num=50, endpoint=True, dtype=None):
        integer_dtype = _integer_torch_dtype(dtype, raw_pytorch, torch)
        if integer_dtype is None:
            return original_linspace(
                start,
                stop,
                num=num,
                endpoint=endpoint,
                dtype=dtype,
            )
        values = original_linspace(start, stop, num=num, endpoint=endpoint, dtype=None)
        return torch.floor(values).to(dtype=integer_dtype)

    linspace.__name__ = getattr(original_linspace, "__name__", "linspace")
    linspace.__doc__ = getattr(original_linspace, "__doc__", None)
    linspace._pyrecest_integer_dtype_contract = True
    raw_pytorch.linspace = linspace
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.linspace = linspace


def _pytorch_creation_dimension(dimension, np) -> int:
    """Return one creation-shape dimension while rejecting booleans."""
    if isinstance(dimension, (bool, np.bool_)):
        raise TypeError("shape dimensions must be integers")
    try:
        return _operator_index(dimension)
    except TypeError as exc:
        raise TypeError("shape dimensions must be integers") from exc


def _pytorch_creation_shape(shape, torch, np) -> tuple[int, ...]:
    """Return a NumPy-style creation shape without accepting boolean dimensions."""
    if torch.is_tensor(shape):
        shape = shape.detach().cpu().numpy()

    if isinstance(shape, (bool, np.bool_)):
        raise TypeError("shape dimensions must be integers")
    if isinstance(shape, (list, tuple)):
        return tuple(_pytorch_creation_dimension(dimension, np) for dimension in shape)

    shape_array = np.asarray(shape)
    if shape_array.shape == ():
        return (_pytorch_creation_dimension(shape_array.item(), np),)
    if shape_array.size and np.issubdtype(shape_array.dtype, np.bool_):
        raise TypeError("shape dimensions must be integers")
    return tuple(
        _pytorch_creation_dimension(dimension, np)
        for dimension in shape_array.tolist()
    )


def _wrap_creation_shape_helper(original_helper, torch, np):
    """Normalize creation shapes before the base PyTorch compatibility wrapper."""
    if getattr(original_helper, "_pyrecest_bool_shape_contract", False):
        return original_helper

    def creation_helper(shape, *args, **kwargs):
        return original_helper(_pytorch_creation_shape(shape, torch, np), *args, **kwargs)

    creation_helper.__name__ = getattr(original_helper, "__name__", "creation_helper")
    creation_helper.__doc__ = getattr(original_helper, "__doc__", None)
    creation_helper._pyrecest_bool_shape_contract = True
    return creation_helper


def _patch_pytorch_creation_bool_shape_contract(raw_pytorch, backend, torch, np) -> None:
    """Reject boolean creation shapes before PyTorch interprets them as integers."""
    helper_names = ("empty", "zeros", "ones", "full")
    if all(
        getattr(getattr(raw_pytorch, helper_name, None), "_pyrecest_bool_shape_contract", False)
        for helper_name in helper_names
    ):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            for helper_name in helper_names:
                setattr(backend, helper_name, getattr(raw_pytorch, helper_name))
        return

    for helper_name in helper_names:
        wrapped_helper = _wrap_creation_shape_helper(
            getattr(raw_pytorch, helper_name),
            torch,
            np,
        )
        setattr(raw_pytorch, helper_name, wrapped_helper)
        if getattr(backend, "__backend_name__", None) == "pytorch":
            setattr(backend, helper_name, wrapped_helper)


def _arraylike_tensor(value, raw_pytorch, torch):
    """Return array-like helper input as a PyTorch tensor."""
    if torch.is_tensor(value):
        return value
    return raw_pytorch.array(value)


def _wrap_arraylike_unary_helper(original_helper, raw_pytorch, torch):
    """Normalize NumPy-style array-like inputs before tensor-only helpers."""
    if getattr(original_helper, "_pyrecest_arraylike_contract", False):
        return original_helper

    def unary_helper(input, *args, **kwargs):  # pylint: disable=redefined-builtin
        return original_helper(
            _arraylike_tensor(input, raw_pytorch, torch),
            *args,
            **kwargs,
        )

    unary_helper.__name__ = getattr(original_helper, "__name__", "unary_helper")
    unary_helper.__doc__ = getattr(original_helper, "__doc__", None)
    unary_helper._pyrecest_arraylike_contract = True
    return unary_helper


def _wrap_argsort_arraylike_helper(original_argsort, raw_pytorch, torch):
    """Normalize NumPy-style array-like inputs before PyTorch argsort."""
    if getattr(original_argsort, "_pyrecest_arraylike_contract", False):
        return original_argsort

    def argsort(input, axis=-1, descending=False, stable=False, *, dim=None):  # pylint: disable=redefined-builtin
        if dim is not None:
            if axis != -1 and axis != dim:
                raise TypeError("argsort() got both 'axis' and 'dim'")
            axis = dim
        return original_argsort(
            _arraylike_tensor(input, raw_pytorch, torch),
            dim=axis,
            descending=descending,
            stable=stable,
        )

    argsort.__name__ = getattr(original_argsort, "__name__", "argsort")
    argsort.__doc__ = getattr(original_argsort, "__doc__", None)
    argsort._pyrecest_arraylike_contract = True
    return argsort


def _patch_pytorch_arraylike_helper_contract(raw_pytorch, backend, torch) -> None:
    """Make tensor-like PyTorch helpers accept NumPy-style array-like inputs."""
    helper_names = (
        "empty_like",
        "ones_like",
        "zeros_like",
        "full_like",
        "isfinite",
        "isinf",
        "isnan",
        "isreal",
    )
    all_helper_names = (*helper_names, "argsort")
    if all(
        getattr(getattr(raw_pytorch, helper_name, None), "_pyrecest_arraylike_contract", False)
        for helper_name in all_helper_names
    ):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            for helper_name in all_helper_names:
                setattr(backend, helper_name, getattr(raw_pytorch, helper_name))
        return

    for helper_name in helper_names:
        wrapped_helper = _wrap_arraylike_unary_helper(
            getattr(raw_pytorch, helper_name),
            raw_pytorch,
            torch,
        )
        setattr(raw_pytorch, helper_name, wrapped_helper)
        if getattr(backend, "__backend_name__", None) == "pytorch":
            setattr(backend, helper_name, wrapped_helper)

    wrapped_argsort = _wrap_argsort_arraylike_helper(raw_pytorch.argsort, raw_pytorch, torch)
    raw_pytorch.argsort = wrapped_argsort
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.argsort = wrapped_argsort


def _patch_pytorch_concatenate_axis_none_contract(raw_pytorch, backend, torch) -> None:
    """Make PyTorch concatenate flatten inputs when axis is ``None``."""
    original_concatenate = raw_pytorch.concatenate
    if getattr(original_concatenate, "_pyrecest_axis_none_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.concatenate = original_concatenate
        return

    def concatenate(seq, axis=0, out=None):
        tensors = [raw_pytorch.array(item) for item in seq]
        if axis is None:
            tensors = [tensor.reshape(-1) for tensor in tensors]
            axis_arg = 0
        else:
            axis_arg = _operator_index(axis)
        tensors = raw_pytorch.convert_to_wider_dtype(tensors)
        return torch.cat(tensors, dim=axis_arg, out=out)

    concatenate.__name__ = getattr(original_concatenate, "__name__", "concatenate")
    concatenate.__doc__ = getattr(original_concatenate, "__doc__", None)
    concatenate._pyrecest_axis_none_contract = True
    raw_pytorch.concatenate = concatenate
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.concatenate = concatenate


__all__ = ["patch_pytorch_dtype_promotion_contract"]
