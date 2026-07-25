# For ffts. Added for pyrecest.
from functools import wraps as _wraps
from operator import index as _operator_index

import torch as _torch

from ._common import array as _array

_BOOLEAN_FFT_LENGTH_ERROR = "n must be an integer length, not boolean"


def _as_fft_tensor(value):
    """Convert array-like FFT inputs to torch tensors."""
    return value if _torch.is_tensor(value) else _array(value)


def _is_boolean_fft_length(value):
    """Return whether a real-FFT length is Boolean-valued."""
    if isinstance(value, bool):
        return True
    if _torch.is_tensor(value):
        return value.dtype == _torch.bool
    dtype = getattr(value, "dtype", None)
    return dtype is not None and str(dtype) in {"bool", "bool_"}


def _validate_real_fft_length_args(args, kwargs):
    """Reject Boolean ``n`` values before PyTorch's FFT argument parser."""
    length = args[0] if args else kwargs.get("n")
    if _is_boolean_fft_length(length):
        raise TypeError(_BOOLEAN_FFT_LENGTH_ERROR)


def _is_empty_dim(dim):
    """Return whether a NumPy-style FFT axis tuple/list selects no axes."""
    if dim is None or isinstance(dim, (str, bytes)):
        return False
    try:
        return len(tuple(dim)) == 0
    except TypeError:
        return False


def _empty_dim_noop_is_valid(args, kwargs):
    """Return whether an empty FFT dimension tuple can safely no-op."""
    shape = args[0] if args else kwargs.get("s", None)
    if shape is None:
        return True
    if isinstance(shape, (str, bytes)):
        return False
    try:
        return len(tuple(shape)) == 0
    except TypeError:
        return False


def _normalize_single_fft_dim(dim):
    """Return scalar-array FFT dimensions as Python integers when possible."""
    if dim is None or isinstance(dim, (bool, str, bytes)):
        return dim
    ndim = getattr(dim, "ndim", None)
    if ndim is not None and ndim != 0:
        return dim
    try:
        return _operator_index(dim)
    except TypeError:
        return dim


def _normalize_fft_dim_sequence(dim):
    """Return NumPy-style FFT dimension sequences in PyTorch-compatible form."""
    if dim is None or isinstance(dim, (bool, str, bytes)):
        return dim
    try:
        return _operator_index(dim)
    except TypeError:
        pass

    try:
        dim_entries = tuple(dim)
    except TypeError:
        return dim

    normalized_entries = []
    for entry in dim_entries:
        if isinstance(entry, bool):
            return dim
        try:
            normalized_entries.append(_operator_index(entry))
        except TypeError:
            return dim
    return tuple(normalized_entries)


def _normalize_fft_shape_sequence(shape):
    """Return NumPy-style FFT shape sequences in PyTorch-compatible form."""
    if shape is None or isinstance(shape, (str, bytes)):
        return shape
    try:
        shape_entries = tuple(shape)
    except TypeError:
        return shape

    normalized_entries = []
    for entry in shape_entries:
        if isinstance(entry, bool):
            normalized_entries.append(entry)
            continue
        try:
            normalized_entries.append(_operator_index(entry))
        except TypeError:
            return shape
    return tuple(normalized_entries)


def _normalize_fft_shape_args(args, kwargs):
    """Normalize the NumPy-style ``s`` FFT shape argument when present."""
    if args:
        return (_normalize_fft_shape_sequence(args[0]), *args[1:]), kwargs
    if "s" not in kwargs:
        return args, kwargs
    kwargs = dict(kwargs)
    kwargs["s"] = _normalize_fft_shape_sequence(kwargs["s"])
    return args, kwargs


def _with_dim_alias(kwargs, alias, func_name, *, none_alias_means_default=True):
    if alias not in kwargs:
        return kwargs

    kwargs = dict(kwargs)
    alias_value = kwargs.pop(alias)
    dim_value = kwargs.get("dim")
    if alias_value is None:
        if none_alias_means_default:
            return kwargs
        if dim_value is not None:
            raise TypeError("conflicting FFT axis aliases")
        kwargs["dim"] = None
        return kwargs
    if dim_value is not None:
        dim_value = _normalize_fft_dim_sequence(dim_value)
        alias_value = _normalize_fft_dim_sequence(alias_value)
        if dim_value != alias_value:
            raise TypeError("conflicting FFT axis aliases")
        return kwargs
    kwargs["dim"] = alias_value
    return kwargs


def _wrap_arraylike_fft(
    torch_func,
    *,
    func_name,
    dim_alias=None,
    empty_dim_is_noop=False,
    normalize_scalar_dim=False,
    normalize_dim_sequence=False,
    normalize_shape_sequence=False,
    validate_real_length=False,
    none_alias_means_default=True,
):
    @_wraps(torch_func)
    def fft_func(value, *args, **kwargs):
        if dim_alias is not None:
            kwargs = _with_dim_alias(
                kwargs,
                dim_alias,
                func_name,
                none_alias_means_default=none_alias_means_default,
            )
        if validate_real_length:
            _validate_real_fft_length_args(args, kwargs)
        if normalize_scalar_dim and "dim" in kwargs:
            kwargs = dict(kwargs)
            kwargs["dim"] = _normalize_single_fft_dim(kwargs["dim"])
        if normalize_dim_sequence and "dim" in kwargs:
            kwargs = dict(kwargs)
            kwargs["dim"] = _normalize_fft_dim_sequence(kwargs["dim"])
        if normalize_shape_sequence:
            args, kwargs = _normalize_fft_shape_args(args, kwargs)
        value = _as_fft_tensor(value)
        if (
            empty_dim_is_noop
            and _is_empty_dim(kwargs.get("dim"))
            and _empty_dim_noop_is_valid(args, kwargs)
        ):
            return value
        return torch_func(value, *args, **kwargs)

    return fft_func


rfft = _wrap_arraylike_fft(
    _torch.fft.rfft,
    func_name="rfft",
    dim_alias="axis",
    normalize_scalar_dim=True,
    validate_real_length=True,
    none_alias_means_default=False,
)
irfft = _wrap_arraylike_fft(
    _torch.fft.irfft,
    func_name="irfft",
    dim_alias="axis",
    normalize_scalar_dim=True,
    validate_real_length=True,
    none_alias_means_default=False,
)
fftshift = _wrap_arraylike_fft(
    _torch.fft.fftshift,
    func_name="fftshift",
    dim_alias="axes",
    empty_dim_is_noop=True,
    normalize_dim_sequence=True,
)
ifftshift = _wrap_arraylike_fft(
    _torch.fft.ifftshift,
    func_name="ifftshift",
    dim_alias="axes",
    empty_dim_is_noop=True,
    normalize_dim_sequence=True,
)
fftn = _wrap_arraylike_fft(
    _torch.fft.fftn,
    func_name="fftn",
    dim_alias="axes",
    empty_dim_is_noop=True,
    normalize_dim_sequence=True,
    normalize_shape_sequence=True,
)
ifftn = _wrap_arraylike_fft(
    _torch.fft.ifftn,
    func_name="ifftn",
    dim_alias="axes",
    empty_dim_is_noop=True,
    normalize_dim_sequence=True,
    normalize_shape_sequence=True,
)
