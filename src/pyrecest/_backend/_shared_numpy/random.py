from ._dispatch import _common
from ._dispatch import numpy as _np

_modify_func_default_dtype = _common._modify_func_default_dtype
_allow_complex_dtype = _common._allow_complex_dtype


rand = _modify_func_default_dtype(
    copy=False, kw_only=True, target=_allow_complex_dtype(target=_np.random.rand)
)

uniform = _modify_func_default_dtype(
    copy=False, kw_only=True, target=_allow_complex_dtype(target=_np.random.uniform)
)


normal = _modify_func_default_dtype(
    copy=False, kw_only=True, target=_allow_complex_dtype(target=_np.random.normal)
)

multivariate_normal = _modify_func_default_dtype(
    copy=False,
    kw_only=True,
    target=_allow_complex_dtype(target=_np.random.multivariate_normal),
)


def choice(a, size=None, replace=True, p=None, axis=0, shuffle=True):
    """Draw samples using NumPy's seeded global random state.

    ``numpy.random.Generator.choice`` supports sampling rows from a multidimensional
    array, but it is independent of ``numpy.random.seed`` when a fresh generator is
    created for every call.  The backend exposes ``random.seed``/``get_state`` from
    ``numpy.random``, so this wrapper samples indices through the seeded legacy RNG
    and then gathers along ``axis`` for multidimensional inputs.
    """
    del shuffle  # ``numpy.random.choice`` has no equivalent shuffle argument.

    a_array = _np.asarray(a)
    if a_array.ndim == 0:
        return _np.random.choice(a, size=size, replace=replace, p=p)

    axis = axis % a_array.ndim
    if a_array.ndim == 1 and axis == 0:
        return _np.random.choice(a_array, size=size, replace=replace, p=p)

    if p is not None:
        p = _np.asarray(p)

    indices = _np.random.choice(a_array.shape[axis], size=size, replace=replace, p=p)
    return _np.take(a_array, indices, axis=axis)
