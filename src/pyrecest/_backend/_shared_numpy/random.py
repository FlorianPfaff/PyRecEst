import operator as _operator

from ._dispatch import _common
from ._dispatch import numpy as _np

_modify_func_default_dtype = _common._modify_func_default_dtype
_allow_complex_dtype = _common._allow_complex_dtype


def _rand(*dims, size=None):
    """Draw uniform samples while accepting the backend ``size=`` contract.

    ``numpy.random.rand`` only accepts legacy positional dimensions, whereas the
    PyRecEst random backend exposes ``rand(size=...)`` like the JAX and PyTorch
    implementations.  Use ``numpy.random.random`` internally so keyword and
    tuple sizes work without dropping support for NumPy's positional form.
    """
    if dims:
        if size is not None:
            raise TypeError("Specify either positional dimensions or size, not both.")
        size = dims[0] if len(dims) == 1 else dims
    return _np.random.random(size)


rand = _modify_func_default_dtype(
    copy=False, kw_only=True, target=_allow_complex_dtype(target=_rand)
)


def _uniform(low=0.0, high=1.0, size=None):
    low_array = _np.asarray(low)
    high_array = _np.asarray(high)
    if _np.any(low_array > high_array):
        raise ValueError("Upper bound must be greater than or equal to lower bound")
    return _np.random.uniform(low, high, size)


uniform = _modify_func_default_dtype(
    copy=False, kw_only=True, target=_allow_complex_dtype(target=_uniform)
)


normal = _modify_func_default_dtype(
    copy=False, kw_only=True, target=_allow_complex_dtype(target=_np.random.normal)
)

multivariate_normal = _modify_func_default_dtype(
    copy=False,
    kw_only=True,
    target=_allow_complex_dtype(target=_np.random.multivariate_normal),
)


def _normalize_choice_axis(axis, ndim):
    if isinstance(axis, (bool, _np.bool_)):
        raise TypeError("axis must be an integer")
    try:
        axis = _operator.index(axis)
    except TypeError as exc:
        raise TypeError("axis must be an integer") from exc
    if axis < -ndim or axis >= ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    return axis % ndim


def _maybe_preserve_choice_order(indices, *, replace, p, shuffle, size):
    if replace or p is not None or bool(shuffle) or size is None:
        return indices

    index_array = _np.asarray(indices)
    if index_array.ndim == 0:
        return indices
    return _np.sort(index_array.reshape(-1)).reshape(index_array.shape)


def choice(a, size=None, replace=True, p=None, axis=0, shuffle=True):
    """Draw samples using NumPy's seeded global random state.

    ``numpy.random.Generator.choice`` supports sampling rows from a multidimensional
    array, but it is independent of ``numpy.random.seed`` when a fresh generator is
    created for every call.  The backend exposes ``random.seed``/``get_state`` from
    ``numpy.random``, so this wrapper samples indices through the seeded legacy RNG
    and then gathers along ``axis`` for multidimensional inputs.
    """

    a_array = _np.asarray(a)
    if a_array.ndim == 0:
        return _maybe_preserve_choice_order(
            _np.random.choice(a, size=size, replace=replace, p=p),
            replace=replace,
            p=p,
            shuffle=shuffle,
            size=size,
        )

    axis = _normalize_choice_axis(axis, a_array.ndim)
    if a_array.ndim == 1 and axis == 0:
        return _maybe_preserve_choice_order(
            _np.random.choice(a_array, size=size, replace=replace, p=p),
            replace=replace,
            p=p,
            shuffle=shuffle,
            size=size,
        )

    if p is not None:
        p = _np.asarray(p)

    indices = _np.random.choice(a_array.shape[axis], size=size, replace=replace, p=p)
    indices = _maybe_preserve_choice_order(
        indices,
        replace=replace,
        p=p,
        shuffle=shuffle,
        size=size,
    )
    return _np.take(a_array, indices, axis=axis)
