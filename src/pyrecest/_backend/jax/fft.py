"""JAX FFT backend wrappers."""

from operator import index as _operator_index

import jax.numpy as _jnp
import numpy as _np
from jax.numpy import fft as _fft

_BOOLEAN_FFT_AXIS_ERROR = "axis must be an integer, not boolean"
_BOOLEAN_FFT_LENGTH_ERROR = "n must be an integer length, not boolean"
_FFT_SHAPE_SEQUENCE_ERROR = "s must be None or a sequence of integer lengths"
_FFT_AXES_SEQUENCE_ERROR = "axes must be None or a sequence of integer axes"


def _normalize_real_fft_axis(axis):
    """Return a Python ``int`` for integer scalar-array FFT axes."""
    if isinstance(axis, (bool, _np.bool_)):
        raise TypeError(_BOOLEAN_FFT_AXIS_ERROR)
    if isinstance(axis, _np.ndarray):
        if _np.issubdtype(axis.dtype, _np.bool_):
            raise TypeError(_BOOLEAN_FFT_AXIS_ERROR)
        if axis.ndim == 0 and _np.issubdtype(axis.dtype, _np.integer):
            return int(axis.item())
        return axis
    if isinstance(axis, _jnp.ndarray):
        axis_dtype = _np.asarray(axis).dtype
        if _np.issubdtype(axis_dtype, _np.bool_):
            raise TypeError(_BOOLEAN_FFT_AXIS_ERROR)
        if axis.ndim == 0 and _np.issubdtype(axis_dtype, _np.integer):
            return int(axis.item())
        return axis
    if isinstance(axis, _np.integer):
        return int(axis)
    return axis


def _normalize_real_fft_length(n):
    """Return a Python ``int`` for integer scalar-array real FFT lengths."""
    if n is None:
        return None
    if isinstance(n, (bool, _np.bool_)):
        raise TypeError(_BOOLEAN_FFT_LENGTH_ERROR)
    if isinstance(n, _np.ndarray):
        if _np.issubdtype(n.dtype, _np.bool_):
            raise TypeError(_BOOLEAN_FFT_LENGTH_ERROR)
        if n.size == 1 and _np.issubdtype(n.dtype, _np.integer):
            return int(n.item())
        return n
    if isinstance(n, _jnp.ndarray):
        n_dtype = _np.asarray(n).dtype
        if _np.issubdtype(n_dtype, _np.bool_):
            raise TypeError(_BOOLEAN_FFT_LENGTH_ERROR)
        if n.size == 1 and _np.issubdtype(n_dtype, _np.integer):
            return int(n.item())
        return n
    if isinstance(n, _np.integer):
        return int(n)
    return n


def _normalize_fft_sequence_item(value, boolean_error, integer_error):
    """Return a Python ``int`` for one FFT shape or axis sequence entry."""
    if isinstance(value, (bool, _np.bool_)):
        raise TypeError(boolean_error)
    if isinstance(value, _np.ndarray):
        if _np.issubdtype(value.dtype, _np.bool_):
            raise TypeError(boolean_error)
        if value.size == 1 and _np.issubdtype(value.dtype, _np.integer):
            return int(value.item())
        raise TypeError(integer_error)
    if isinstance(value, _jnp.ndarray):
        value_array = _np.asarray(value)
        if _np.issubdtype(value_array.dtype, _np.bool_):
            raise TypeError(boolean_error)
        if value_array.size == 1 and _np.issubdtype(value_array.dtype, _np.integer):
            return int(value_array.item())
        raise TypeError(integer_error)
    try:
        return _operator_index(value)
    except TypeError as exc:
        raise TypeError(integer_error) from exc


def _normalize_fft_integer_sequence(
    value, sequence_error, boolean_error, integer_error
):
    """Normalize NumPy/JAX scalar-array entries inside FFT integer sequences."""
    if value is None:
        return None
    if isinstance(value, (bool, _np.bool_)):
        raise TypeError(sequence_error)
    if isinstance(value, _np.ndarray):
        if _np.issubdtype(value.dtype, _np.bool_):
            raise TypeError(boolean_error)
        if value.ndim == 0:
            raise TypeError(sequence_error)
        return tuple(
            _normalize_fft_sequence_item(item, boolean_error, integer_error)
            for item in value.tolist()
        )
    if isinstance(value, _jnp.ndarray):
        value_array = _np.asarray(value)
        if _np.issubdtype(value_array.dtype, _np.bool_):
            raise TypeError(boolean_error)
        if value_array.ndim == 0:
            raise TypeError(sequence_error)
        return tuple(
            _normalize_fft_sequence_item(item, boolean_error, integer_error)
            for item in value_array.tolist()
        )
    if isinstance(value, (str, bytes)):
        raise TypeError(sequence_error)
    try:
        entries = tuple(value)
    except TypeError as exc:
        raise TypeError(sequence_error) from exc
    return tuple(
        _normalize_fft_sequence_item(item, boolean_error, integer_error)
        for item in entries
    )


def _normalize_complex_fft_shape(s):
    return _normalize_fft_integer_sequence(
        s,
        _FFT_SHAPE_SEQUENCE_ERROR,
        "s entries must be integer lengths, not boolean",
        "s entries must be integer lengths",
    )


def _normalize_complex_fft_axes(axes):
    return _normalize_fft_integer_sequence(
        axes,
        _FFT_AXES_SEQUENCE_ERROR,
        "axes entries must be integers, not boolean",
        "axes entries must be integers",
    )


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
    return _fft.fftn(
        _jnp.asarray(a),
        s=_normalize_complex_fft_shape(s),
        axes=_normalize_complex_fft_axes(axes),
        norm=norm,
    )


def ifftn(a, s=None, axes=None, norm=None):
    return _fft.ifftn(
        _jnp.asarray(a),
        s=_normalize_complex_fft_shape(s),
        axes=_normalize_complex_fft_axes(axes),
        norm=norm,
    )


def fftshift(x, axes=None):
    return _fft.fftshift(_jnp.asarray(x), axes=axes)


def ifftshift(x, axes=None):
    return _fft.ifftshift(_jnp.asarray(x), axes=axes)
