import operator as _operator

from ._dispatch import _common
from ._dispatch import numpy as _np

_modify_func_default_dtype = _common._modify_func_default_dtype
_allow_complex_dtype = _common._allow_complex_dtype
_BOOLEAN_TYPES = (bool, _np.bool_)
_TEMPORAL_DTYPE_KINDS = "Mm"


def _rand(*dims, size=None):
    """Draw uniform samples with NumPy-style positional and size arguments."""
    if dims:
        if size is not None:
            raise TypeError("Specify either positional dimensions or size, not both.")
        size = dims[0] if len(dims) == 1 else dims
    return _np.random.random(_normalize_size(size))


rand = _modify_func_default_dtype(
    copy=False, kw_only=True, target=_allow_complex_dtype(target=_rand)
)


def _contains_boolean_value(value):
    if isinstance(value, _BOOLEAN_TYPES):
        return True
    try:
        values = _np.asarray(value, dtype=object).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return False
    return any(isinstance(item, _BOOLEAN_TYPES) for item in values)


def _is_temporal_scalar_array(value_array):
    return value_array.ndim == 0 and value_array.dtype.kind in _TEMPORAL_DTYPE_KINDS


def _size_type_error():
    return TypeError("size must be None, an integer, or a sequence of integers")


def _normalize_size(size):
    if size is None:
        return None
    if _contains_boolean_value(size) or isinstance(size, (str, bytes)):
        raise _size_type_error()
    try:
        size_array = _np.asarray(size)
    except (TypeError, ValueError) as exc:
        raise _size_type_error() from exc
    if size_array.ndim == 1 and size_array.size == 0:
        return ()
    if size_array.dtype.kind not in "iu":
        raise _size_type_error()
    if size_array.ndim == 0:
        dimension = int(size_array.item())
        if dimension < 0:
            raise ValueError("size dimensions must be non-negative")
        return dimension
    if size_array.ndim != 1:
        raise _size_type_error()
    dimensions = tuple(int(dimension) for dimension in size_array.tolist())
    if any(dimension < 0 for dimension in dimensions):
        raise ValueError("size dimensions must be non-negative")
    return dimensions


def _validate_uniform_bound(bound, name):
    if _contains_boolean_value(bound):
        raise TypeError(f"{name} must be real numeric, not boolean")
    try:
        bound_array = _np.asarray(bound)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be real numeric") from exc
    if bound_array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be real numeric")
    if _np.any(~_np.isfinite(bound_array)):
        raise ValueError("uniform bounds must be finite")
    return bound_array


def _validate_uniform_bounds(low, high):
    low_array = _validate_uniform_bound(low, "low")
    high_array = _validate_uniform_bound(high, "high")
    if _np.any(low_array > high_array):
        raise ValueError("Upper bound must be greater than or equal to lower bound")


def _uniform(low=0.0, high=1.0, size=None):
    _validate_uniform_bounds(low, high)
    return _np.random.uniform(low, high, _normalize_size(size))


uniform = _modify_func_default_dtype(
    copy=False, kw_only=True, target=_allow_complex_dtype(target=_uniform)
)


def _validate_normal_parameter(value, name):
    if _contains_boolean_value(value):
        raise TypeError(f"{name} must be real numeric, not boolean")
    try:
        value_array = _np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be real numeric") from exc
    if value_array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be real numeric")
    if _np.any(~_np.isfinite(value_array)):
        raise ValueError(f"{name} must be finite")
    return value_array


def _validate_normal_scale(scale):
    scale_array = _validate_normal_parameter(scale, "scale")
    if _np.any(scale_array < 0):
        raise ValueError("scale must be non-negative")
    return scale_array


def _normal(loc=0.0, scale=1.0, size=None):
    loc = _validate_normal_parameter(loc, "loc")
    scale = _validate_normal_scale(scale)
    return _np.random.normal(loc, scale, _normalize_size(size))


normal = _modify_func_default_dtype(
    copy=False, kw_only=True, target=_allow_complex_dtype(target=_normal)
)


def _validate_multivariate_normal_parameter(value, name):
    if _contains_boolean_value(value):
        raise TypeError(f"{name} must be real numeric, not boolean")
    try:
        value_array = _np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be real numeric") from exc
    if value_array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be real numeric")
    if _np.any(~_np.isfinite(value_array)):
        raise ValueError(f"{name} must be finite")
    return value_array


def _multivariate_normal(mean, cov, size=None, check_valid="warn", tol=1e-8):
    mean = _validate_multivariate_normal_parameter(mean, "mean")
    cov = _validate_multivariate_normal_parameter(cov, "cov")
    return _np.random.multivariate_normal(
        mean, cov, _normalize_size(size), check_valid, tol
    )


multivariate_normal = _modify_func_default_dtype(
    copy=False,
    kw_only=True,
    target=_allow_complex_dtype(target=_multivariate_normal),
)


def _normalize_choice_axis(axis, ndim):
    if isinstance(axis, _BOOLEAN_TYPES):
        raise TypeError("axis must be an integer")
    try:
        axis = _operator.index(axis)
    except TypeError as exc:
        raise TypeError("axis must be an integer") from exc
    if axis < -ndim or axis >= ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    return axis % ndim


def _choice_bool(value, name):
    if isinstance(value, _BOOLEAN_TYPES):
        return bool(value)
    value_array = _np.asarray(value)
    if value_array.shape == () and value_array.dtype.kind == "b":
        return bool(value_array.item())
    raise TypeError(f"{name} must be a boolean")


def _validate_choice_population(a_array):
    if a_array.ndim != 0:
        return
    if _is_temporal_scalar_array(a_array):
        raise ValueError("a must be a positive integer or a non-empty array")
    scalar = a_array.item()
    if isinstance(scalar, _BOOLEAN_TYPES):
        raise ValueError("a must be a positive integer or a non-empty array")


def _normalize_probability_values(values):
    if _np.any(values < 0.0) or _np.any(~_np.isfinite(values)):
        raise ValueError("probabilities do not sum to a positive value")
    scale = float(values.max()) if values.size else 0.0
    if scale <= 0.0:
        raise ValueError("probabilities do not sum to a positive value")
    scaled = values / scale
    total = scaled.sum()
    if not _np.isfinite(total) or total <= 0.0:
        raise ValueError("probabilities do not sum to a positive value")
    return scaled / total


def _maybe_preserve_choice_order(indices, *, replace, p, shuffle, size):
    if replace or p is not None or shuffle or size is None:
        return indices

    index_array = _np.asarray(indices)
    if index_array.ndim == 0:
        return indices
    return _np.sort(index_array.reshape(-1)).reshape(index_array.shape)


def _integer_choice_population_size(a_array):
    if a_array.ndim == 0:
        if _is_temporal_scalar_array(a_array):
            raise ValueError("a must be a positive integer or a non-empty array")
        try:
            return _operator.index(a_array.item())
        except TypeError:
            return None
    return None


def _validate_choice_probabilities(p, population_size):
    if p is None:
        return None
    if _contains_boolean_value(p):
        raise TypeError("p must be real numeric, not boolean")
    try:
        p_array = _np.asarray(p)
    except (TypeError, ValueError) as exc:
        raise TypeError("p must be real numeric") from exc
    if p_array.dtype.kind not in "iuf":
        raise TypeError("p must be real numeric")
    p_array = p_array.astype(_np.float64, copy=False)
    if p_array.ndim != 1 or p_array.shape[0] != population_size:
        raise ValueError("p must be 1-dimensional with one entry per population item")
    return _normalize_probability_values(p_array)


def choice(a, size=None, replace=True, p=None, axis=0, shuffle=True):
    """Draw samples from an integer or array population."""
    replace = _choice_bool(replace, "replace")
    shuffle = _choice_bool(shuffle, "shuffle")
    size = _normalize_size(size)
    a_array = _np.asarray(a)
    _validate_choice_population(a_array)
    if a_array.ndim == 0:
        population_size = _integer_choice_population_size(a_array)
        if population_size is not None:
            p = _validate_choice_probabilities(p, population_size)
        return _maybe_preserve_choice_order(
            _np.random.choice(a, size=size, replace=replace, p=p),
            replace=replace,
            p=p,
            shuffle=shuffle,
            size=size,
        )

    axis = _normalize_choice_axis(axis, a_array.ndim)
    p = _validate_choice_probabilities(p, a_array.shape[axis])
    if a_array.ndim == 1 and axis == 0:
        indices = _np.random.choice(a_array.shape[0], size=size, replace=replace, p=p)
        indices = _maybe_preserve_choice_order(
            indices,
            replace=replace,
            p=p,
            shuffle=shuffle,
            size=size,
        )
        return _np.take(a_array, indices, axis=0)

    indices = _np.random.choice(a_array.shape[axis], size=size, replace=replace, p=p)
    indices = _maybe_preserve_choice_order(
        indices,
        replace=replace,
        p=p,
        shuffle=shuffle,
        size=size,
    )
    return _np.take(a_array, indices, axis=axis)
