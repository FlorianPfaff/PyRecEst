# For ffts. Added for pyrecest.
import torch as _torch


def _as_tensor(x):
    return x if _torch.is_tensor(x) else _torch.as_tensor(x)


def _resolve_dim_alias(dim, alias, alias_name, func_name, *, default=None):
    if alias is None:
        return default if dim is None else dim
    if dim is not None and dim != alias:
        raise TypeError(f"{func_name}() got both 'dim' and '{alias_name}'")
    return alias


def fftn(input, s=None, dim=None, norm=None, *, axes=None, out=None):
    dim = _resolve_dim_alias(dim, axes, "axes", "fftn")
    return _torch.fft.fftn(_as_tensor(input), s=s, dim=dim, norm=norm, out=out)


def ifftn(input, s=None, dim=None, norm=None, *, axes=None, out=None):
    dim = _resolve_dim_alias(dim, axes, "axes", "ifftn")
    return _torch.fft.ifftn(_as_tensor(input), s=s, dim=dim, norm=norm, out=out)


def rfft(input, n=None, dim=None, norm=None, *, axis=None, out=None):
    dim = _resolve_dim_alias(dim, axis, "axis", "rfft", default=-1)
    return _torch.fft.rfft(_as_tensor(input), n=n, dim=dim, norm=norm, out=out)


def irfft(input, n=None, dim=None, norm=None, *, axis=None, out=None):
    dim = _resolve_dim_alias(dim, axis, "axis", "irfft", default=-1)
    return _torch.fft.irfft(_as_tensor(input), n=n, dim=dim, norm=norm, out=out)


def fftshift(input, dim=None, *, axes=None):
    dim = _resolve_dim_alias(dim, axes, "axes", "fftshift")
    return _torch.fft.fftshift(_as_tensor(input), dim=dim)


def ifftshift(input, dim=None, *, axes=None):
    dim = _resolve_dim_alias(dim, axes, "axes", "ifftshift")
    return _torch.fft.ifftshift(_as_tensor(input), dim=dim)
