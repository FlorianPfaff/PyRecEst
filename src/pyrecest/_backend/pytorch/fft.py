# For ffts. Added for pyrecest.
import torch as _torch

from ._common import array


def _as_fft_tensor(x):
    return x if _torch.is_tensor(x) else array(x)


def _resolve_axis(axis, dim, func_name):
    if dim is not None:
        if axis != -1 and axis != dim:
            raise TypeError(func_name + "() got both 'axis' and 'dim'")
        axis = dim
    return axis


def _resolve_axes(axes, dim, func_name):
    if dim is not None:
        if axes is not None and axes != dim:
            raise TypeError(func_name + "() got both 'axes' and 'dim'")
        axes = dim
    return axes


def rfft(a, n=None, axis=-1, norm=None, *, dim=None):
    """Compute a real FFT after coercing array-like inputs."""
    axis = _resolve_axis(axis, dim, "rfft")
    return _torch.fft.rfft(_as_fft_tensor(a), n=n, dim=axis, norm=norm)


def irfft(a, n=None, axis=-1, norm=None, *, dim=None):
    """Compute an inverse real FFT after coercing array-like inputs."""
    axis = _resolve_axis(axis, dim, "irfft")
    return _torch.fft.irfft(_as_fft_tensor(a), n=n, dim=axis, norm=norm)


def fftn(a, s=None, axes=None, norm=None, *, dim=None):
    """Compute an N-D FFT after coercing array-like inputs."""
    axes = _resolve_axes(axes, dim, "fftn")
    return _torch.fft.fftn(_as_fft_tensor(a), s=s, dim=axes, norm=norm)


def ifftn(a, s=None, axes=None, norm=None, *, dim=None):
    """Compute an inverse N-D FFT after coercing array-like inputs."""
    axes = _resolve_axes(axes, dim, "ifftn")
    return _torch.fft.ifftn(_as_fft_tensor(a), s=s, dim=axes, norm=norm)


def fftshift(x, axes=None, *, dim=None):
    """Shift zero-frequency components after coercing array-like inputs."""
    axes = _resolve_axes(axes, dim, "fftshift")
    return _torch.fft.fftshift(_as_fft_tensor(x), dim=axes)


def ifftshift(x, axes=None, *, dim=None):
    """Inverse-shift zero-frequency components after coercing array-like inputs."""
    axes = _resolve_axes(axes, dim, "ifftshift")
    return _torch.fft.ifftshift(_as_fft_tensor(x), dim=axes)
