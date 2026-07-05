"""JAX FFT backend wrappers."""

import jax.numpy as _jnp
import numpy as _np
from jax.numpy import fft as _fft


def _normalize_real_fft_axis(axis):
    """Return a Python ``int`` for integer scalar-array FFT axes."""
    if isinstance(axis, _np.ndarray):
        if (
            axis.size == 1
            and _np.issubdtype(axis.dtype, _np.integer)
            and not _np.issubdtype(axis.dtype, _np.bool_)
        ):
            return int(axis.item())
        return axis
    if isinstance(axis, _jnp.ndarray):
        axis_dtype = _np.asarray(axis).dtype
        if (
            axis.ndim == 0
            and _np.issubdtype(axis_dtype, _np.integer)
            and not _np.issubdtype(axis_dtype, _np.bool_)
        ):
            return int(axis.item())
        return axis
    if isinstance(axis, _np.integer) and not isinstance(axis, _np.bool_):
        return int(axis)
    return axis


def _normalize_fft_axes(axes):
    """Return Python ``int`` tuples for integer array FFT axis sequences."""
    if axes is None:
        return None
    if isinstance(axes, _jnp.ndarray):
        axes_array = _np.asarray(axes)
    elif isinstance(axes, _np.ndarray):
        axes_array = axes
    else:
        return axes

    if axes_array.ndim == 0:
        return axes
    if axes_array.ndim == 1 and _np.issubdtype(
        axes_array.dtype, _np.integer
    ) and not _np.issubdtype(axes_array.dtype, _np.bool_):
        return tuple(int(axis) for axis in axes_array.tolist())
    return axes


def rfft(a, n=None, axis=-1, norm=None):
    return _fft.rfft(_jnp.asarray(a), n=n, axis=_normalize_real_fft_axis(axis), norm=norm)


def irfft(a, n=None, axis=-1, norm=None):
    return _fft.irfft(_jnp.asarray(a), n=n, axis=_normalize_real_fft_axis(axis), norm=norm)


def fftn(a, s=None, axes=None, norm=None):
    return _fft.fftn(_jnp.asarray(a), s=s, axes=_normalize_fft_axes(axes), norm=norm)


def ifftn(a, s=None, axes=None, norm=None):
    return _fft.ifftn(_jnp.asarray(a), s=s, axes=_normalize_fft_axes(axes), norm=norm)


def fftshift(x, axes=None):
    return _fft.fftshift(_jnp.asarray(x), axes=_normalize_fft_axes(axes))


def ifftshift(x, axes=None):
    return _fft.ifftshift(_jnp.asarray(x), axes=_normalize_fft_axes(axes))
