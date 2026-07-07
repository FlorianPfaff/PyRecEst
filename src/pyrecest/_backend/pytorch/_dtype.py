import functools

import numpy as _np
import torch as _torch
from pyrecest._backend import _backend_config as _config
from pyrecest._backend._dtype_utils import (
    _modify_func_default_dtype,
    _pre_allow_complex_dtype,
    _pre_cast_out_to_input_dtype,
    _update_default_dtypes,
)
from pyrecest._backend._dtype_utils import (
    get_default_cdtype as _shared_get_default_cdtype,
)
from pyrecest._backend._dtype_utils import (
    get_default_dtype as _shared_get_default_dtype,
)
from torch import bool as torch_bool
from torch import (
    complex64,
    complex128,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
)

from ._common import cast

MAP_DTYPE = {
    "bool": torch_bool,
    "uint8": uint8,
    "int8": int8,
    "int16": int16,
    "int32": int32,
    "int64": int64,
    "float16": float16,
    "float32": float32,
    "float64": float64,
    "complex64": complex64,
    "complex128": complex128,
}

_FLOAT_TO_COMPLEX_DTYPE = {
    float32: complex64,
    float64: complex128,
}

_COMPLEX_DTYPES = (complex64, complex128)


def _dtype_key(value):
    """Return the canonical PyTorch dtype-map key for dtype-like values."""
    try:
        return str(_np.dtype(value))
    except (TypeError, ValueError):
        text = str(value)
        if text.startswith("torch."):
            return text.split(".")[-1]
        if text.endswith("'>") and "." in text:
            return text.rsplit(".", maxsplit=1)[-1].removesuffix("'>")
        return text


def _normalize_torch_dtype(dtype, *, default):
    """Return a torch dtype for dtype-like values from other backends."""
    if dtype is None:
        return default
    if isinstance(dtype, _torch.dtype):
        return dtype
    try:
        return MAP_DTYPE[_dtype_key(dtype)]
    except KeyError:
        return dtype


def get_default_dtype():
    """Get the PyTorch backend default floating dtype."""
    return _normalize_torch_dtype(_shared_get_default_dtype(), default=float64)


def get_default_cdtype():
    """Get the PyTorch backend default complex dtype."""
    return _normalize_torch_dtype(_shared_get_default_cdtype(), default=complex128)


def is_floating(x):
    return x.dtype.is_floating_point


def is_complex(x):
    return x.dtype.is_complex


def is_bool(x):
    return x.dtype is _torch.bool


def as_dtype(value):
    """Transform string or dtype-like value into a PyTorch dtype."""
    return MAP_DTYPE[_dtype_key(value)]


def _dtype_as_str(dtype):
    return str(dtype).split(".")[-1]


def set_default_dtype(value):
    """Set backend default dtype.

    Parameters
    ----------
    value : str or dtype-like
        Floating dtype alias resolving to ``float32`` or ``float64``.
    """
    dtype = _normalize_torch_dtype(value, default=None)
    try:
        complex_dtype = _FLOAT_TO_COMPLEX_DTYPE[dtype]
    except KeyError as exc:
        raise ValueError(
            "PyTorch default dtype must resolve to torch.float32 or torch.float64."
        ) from exc

    _config.DEFAULT_DTYPE = dtype
    _config.DEFAULT_COMPLEX_DTYPE = complex_dtype
    _torch.set_default_dtype(_config.DEFAULT_DTYPE)

    _update_default_dtypes()

    return _config.DEFAULT_DTYPE


def _add_default_dtype_by_casting(target=None):
    """Add the PyTorch default dtype to functions by casting output."""

    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(*args, dtype=None, **kwargs):
            dtype = _normalize_torch_dtype(dtype, default=get_default_dtype())

            out = func(*args, **kwargs)
            if out.dtype != dtype:
                return cast(out, dtype)
            return out

        return _wrapped

    if target is None:
        return _decorator

    return _decorator(target)


_cast_out_to_input_dtype = _pre_cast_out_to_input_dtype(
    cast, is_floating, is_complex, as_dtype, _dtype_as_str
)
_base_allow_complex_dtype = _pre_allow_complex_dtype(cast, _COMPLEX_DTYPES)


def _is_random_array_parameter(value):
    return (
        _torch.is_tensor(value)
        or isinstance(value, (list, tuple))
        or (not isinstance(value, (str, bytes)) and hasattr(value, "__array__"))
    )


def _is_binary_array_operand(value):
    return isinstance(value, (list, tuple)) or (
        not isinstance(value, (str, bytes)) and hasattr(value, "__array__")
    )


def _binary_tensor_operand(value, reference):
    kwargs = {}
    if _torch.is_tensor(reference):
        kwargs["device"] = reference.device
    return _torch.tensor(value, **kwargs)


def _is_real_numeric_dtype(dtype):
    return dtype.is_floating_point or dtype in {
        _torch.uint8,
        _torch.int8,
        _torch.int16,
        _torch.int32,
        _torch.int64,
    }


def _normal_scale_from_call(args, kwargs):
    if "scale" in kwargs:
        return kwargs["scale"]
    if len(args) >= 2:
        return args[1]
    return 1.0


def _validate_scalar_normal_scale(scale):
    if _is_random_array_parameter(scale):
        return

    try:
        scale = _torch.as_tensor(scale)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError("scale must be real numeric") from exc
    if scale.ndim != 0 or not _is_real_numeric_dtype(scale.dtype):
        raise TypeError("scale must be real numeric")
    if bool(scale < 0):
        raise ValueError("scale must be non-negative")


def _allow_complex_dtype(target=None):
    def _decorator(func):
        wrapped = _base_allow_complex_dtype(func)
        if getattr(func, "__name__", "") != "normal":
            return wrapped

        @functools.wraps(wrapped)
        def _wrapped(*args, **kwargs):
            _validate_scalar_normal_scale(_normal_scale_from_call(args, kwargs))
            return wrapped(*args, **kwargs)

        return _wrapped

    if target is None:
        return _decorator

    return _decorator(target)


def _preserve_input_dtype(target=None):
    """Ensure input dtype is preserved.

    How it works?
    -------------
    Only acts on input. Assumes dtype is kwarg and function accepts dtype.
    Passes dtype as input dtype.
    Use together with `_add_default_dtype_by_casting`.
    """

    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(x, *args, dtype=None, **kwargs):
            if dtype is None:
                dtype = x.dtype
            else:
                dtype = _normalize_torch_dtype(dtype, default=get_default_dtype())

            return func(x, *args, dtype=dtype, **kwargs)

        return _wrapped

    if target is None:
        return _decorator

    return _decorator(target)


def _box_unary_scalar(target=None):
    """Update dtype if input is float in unary operations.

    How it works?
    -------------
    Promotes input to tensor if not the case.
    """

    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(x, *args, **kwargs):
            if not _torch.is_tensor(x):
                x = _torch.tensor(x)
            return func(x, *args, **kwargs)

        return _wrapped

    if target is None:
        return _decorator

    return _decorator(target)


def _box_binary_scalar(target=None, box_x1=True, box_x2=True):
    """Update dtype if input is float in binary operations.

    How it works?
    -------------
    Promotes inputs to tensor if not the case.
    """

    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(x1, x2, *args, **kwargs):
            if box_x1 and not _torch.is_tensor(x1):
                x1 = _torch.tensor(x1)
            if box_x2 and not _torch.is_tensor(x2):
                x2 = _binary_tensor_operand(x2, x1)
            elif (
                not box_x2 and not _torch.is_tensor(x2) and _is_binary_array_operand(x2)
            ):
                x2 = _binary_tensor_operand(x2, x1)

            return func(x1, x2, *args, **kwargs)

        return _wrapped

    if target is None:
        return _decorator

    return _decorator(target)


def _patch_parent_log1p_arraylike_contract() -> None:
    """Make PyTorch backend ``log1p`` accept NumPy-style array-like inputs."""
    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - parent backend import failed earlier
        return

    original_log1p = getattr(pytorch_backend, "log1p", None)
    if original_log1p is None or getattr(
        original_log1p, "_pyrecest_arraylike_contract", False
    ):
        return

    log1p = _box_unary_scalar(target=_torch.log1p)
    log1p.__name__ = getattr(original_log1p, "__name__", "log1p")
    log1p.__doc__ = getattr(original_log1p, "__doc__", None)
    log1p._pyrecest_arraylike_contract = True
    pytorch_backend.log1p = log1p


_patch_parent_log1p_arraylike_contract()
