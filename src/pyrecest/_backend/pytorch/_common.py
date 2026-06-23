import numpy as _np
import torch as _torch

_TORCH_DTYPE_BY_NAME = {
    "bool": _torch.bool,
    "uint8": _torch.uint8,
    "int8": _torch.int8,
    "int16": _torch.int16,
    "int32": _torch.int32,
    "int64": _torch.int64,
    "float16": _torch.float16,
    "float32": _torch.float32,
    "float64": _torch.float64,
    "complex64": _torch.complex64,
    "complex128": _torch.complex128,
}


def _normalize_dtype(dtype):
    """Return a torch dtype for dtype-like values understood by NumPy."""
    if dtype is None or isinstance(dtype, _torch.dtype):
        return dtype
    try:
        return _TORCH_DTYPE_BY_NAME[str(_np.dtype(dtype))]
    except (KeyError, TypeError):
        return dtype


def from_numpy(x):
    if _torch.is_tensor(x):
        return x
    if isinstance(x, _np.ndarray) and any(stride < 0 for stride in x.strides):
        x = x.copy()
    return _torch.from_numpy(x)


def array(val, dtype=None):
    dtype = _normalize_dtype(dtype)
    if _torch.is_tensor(val):
        if dtype is None or val.dtype == dtype:
            return val.clone()

        return cast(val, dtype=dtype)

    if isinstance(val, _np.ndarray):
        tensor = from_numpy(val)
        if dtype is not None and tensor.dtype != dtype:
            tensor = cast(tensor, dtype=dtype)

        return tensor

    if isinstance(val, (list, tuple)) and len(val):
        tensors = [array(tensor, dtype=dtype) for tensor in val]
        return _torch.stack(tensors)

    return _torch.tensor(val, dtype=dtype)


def cast(x, dtype):
    dtype = _normalize_dtype(dtype)
    if _torch.is_tensor(x):
        return x.to(dtype=dtype)
    return array(x, dtype=dtype)
