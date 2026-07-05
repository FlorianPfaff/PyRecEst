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


def _normalize_real_fft_length(n):
    """Return a Python ``int`` for integer scalar-array real FFT lengths."""
    if n is None:
        return None
    if isinstance(n, _np.ndarray):
        if (
            n.size == 1
            and _np.issubdtype(n.dtype, _np.integer)
            and not _np.issubdtype(n.dtype, _np.bool_)
        ):
            return int(n.item())
        return n
    if isinstance(n, _jnp.ndarray):
        n_dtype = _np.asarray(n).dtype
        if (
            n.size == 1
            and _np.issubdtype(n_dtype, _np.integer)
            and not _np.issubdtype(n_dtype, _np.bool_)
        ):
            return int(n.item())
        return n
    if isinstance(n, _np.integer) and not isinstance(n, _np.bool_):
        return int(n)
    return n


def rfft(a, n=None, axis=-1, norm=None):
    return _fft.rfft(
        _jnp.asarray(a),
        n=_normalize_real_fft_length(n),
        axis=_normalize_real_fft_axis(axis),
        norm=norm,
    )


def irfft(a, n=None, axis=-1, norm=None):
    return _fft.irfft(
        _jnp.asarray(a),
        n=_normalize_real_fft_length(n),
        axis=_normalize_real_fft_axis(axis),
        norm=norm,
    )


def fftn(a, s=None, axes=None, norm=None):
    return _fft.fftn(_jnp.asarray(a), s=s, axes=axes, norm=norm)


def ifftn(a, s=None, axes=None, norm=None):
    return _fft.ifftn(_jnp.asarray(a), s=s, axes=axes, norm=norm)


def fftshift(x, axes=None):
    return _fft.fftshift(_jnp.asarray(x), axes=axes)


def ifftshift(x, axes=None):
    return _fft.ifftshift(_jnp.asarray(x), axes=axes)
