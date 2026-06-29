# For ffts. Added for pyrecest.
from functools import wraps as _wraps

import torch as _torch

from ._common import array as _array


def _as_fft_tensor(value):
    """Convert array-like FFT inputs to torch tensors."""
    return value if _torch.is_tensor(value) else _array(value)


def _with_dim_alias(kwargs, alias, func_name):
    if alias not in kwargs:
        return kwargs

    kwargs = dict(kwargs)
    alias_value = kwargs.pop(alias)
    if alias_value is None:
        return kwargs
    dim_value = kwargs.get("dim")
    if dim_value is not None:
        if alias_value is not None and dim_value != alias_value:
            raise TypeError("conflicting FFT axis aliases")
        return kwargs
    kwargs["dim"] = alias_value
    return kwargs


def _wrap_arraylike_fft(torch_func, *, func_name, dim_alias=None):
    @_wraps(torch_func)
    def fft_func(value, *args, **kwargs):
        if dim_alias is not None:
            kwargs = _with_dim_alias(kwargs, dim_alias, func_name)
        return torch_func(_as_fft_tensor(value), *args, **kwargs)

    return fft_func


rfft = _wrap_arraylike_fft(_torch.fft.rfft, func_name="rfft", dim_alias="axis")
irfft = _wrap_arraylike_fft(_torch.fft.irfft, func_name="irfft", dim_alias="axis")
fftshift = _wrap_arraylike_fft(
    _torch.fft.fftshift, func_name="fftshift", dim_alias="axes"
)
ifftshift = _wrap_arraylike_fft(
    _torch.fft.ifftshift, func_name="ifftshift", dim_alias="axes"
)
fftn = _wrap_arraylike_fft(_torch.fft.fftn, func_name="fftn", dim_alias="axes")
ifftn = _wrap_arraylike_fft(_torch.fft.ifftn, func_name="ifftn", dim_alias="axes")
