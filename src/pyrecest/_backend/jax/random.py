"""
Jax-based random backend.
Based on random.py by emilemathieu on
https://github.com/oxcsml/geomstats/blob/master/geomstats/_backend/jax/random.py
who says he was in inspired by https://github.com/wesselb/lab/blob/master/lab/jax/random.py
"""

import sys
from operator import index as _operator_index

import jax
import jax.numpy as _jnp
import numpy as _np

backend = sys.modules[__name__]
_BOOLEAN_TYPES = (bool, _np.bool_)


def create_random_state(seed=0):
    return jax.random.PRNGKey(seed=seed)


def seed(seed=None):
    """Reset both the NumPy and JAX RNG state used by this backend."""
    if seed is None:
        seed = int(_np.random.SeedSequence().generate_state(1, dtype=_np.uint32)[0])

    seed_sequence = _np.random.SeedSequence(seed)
    jax_seed = int(seed_sequence.generate_state(1, dtype=_np.uint32)[0])
    backend.jax_global_random_state = create_random_state(jax_seed)
    _np.random.seed(seed)


backend.jax_global_random_state = jax.random.PRNGKey(seed=0)


def global_random_state():
    return backend.jax_global_random_state


def set_global_random_state(state):
    backend.jax_global_random_state = state


get_state = global_random_state
set_state = set_global_random_state


def _get_state(**kwargs):
    has_state = "state" in kwargs
    state = kwargs.pop("state", backend.jax_global_random_state)
    return state, has_state, kwargs


def _scalar_integer_dimension(value):
    if isinstance(value, (bool, _np.bool_)):
        return None
    if isinstance(value, (int, _np.integer)):
        return int(value)
    if hasattr(value, "ndim") and value.ndim == 0:
        value_array = _np.asarray(value)
        if _np.issubdtype(value_array.dtype, _np.bool_):
            return None
        if _np.issubdtype(value_array.dtype, _np.integer):
            return int(value_array.item())
    return None


def _looks_like_integer_dimension(value):
    return _scalar_integer_dimension(value) is not None


def _size_type_error():
    return TypeError("size must be None, an integer, or a sequence of integers")


def _integer_dimension(value):
    value = _scalar_integer_dimension(value)
    if value is None:
        raise _size_type_error()
    if value < 0:
        raise ValueError("size dimensions must be non-negative")
    return value


def _shape_from_size(size):
    """Convert a NumPy-style ``size`` argument to JAX's shape argument."""
    if size is None:
        return ()
    if _looks_like_integer_dimension(size):
        return (_integer_dimension(size),)
    if isinstance(size, (str, bytes)) or not hasattr(size, "__iter__"):
        raise _size_type_error()
    return tuple(_integer_dimension(dim) for dim in size)


def _shape_has_no_samples(shape):
    return any(dim == 0 for dim in shape)


def _shape_from_rand_args(dims, size):
    """Convert NumPy-style ``rand`` dimensions and ``size=`` to a shape."""
    if dims:
        if size is not None:
            raise TypeError("Specify either positional dimensions or size, not both.")
        size = dims[0] if len(dims) == 1 else dims
    return _shape_from_size(size)


def _broadcast_shape_from_values(*values):
    return _np.broadcast_shapes(*(tuple(value.shape) for value in values))


def _bounded_sampler_shape(size, *parameters):
    """Return the NumPy-compatible output shape for bounded random samplers."""
    try:
        parameter_shape = _broadcast_shape_from_values(*parameters)
    except ValueError as exc:
        raise ValueError("parameter arrays could not be broadcast together") from exc

    if size is None:
        return parameter_shape

    sample_shape = _shape_from_size(size)
    try:
        broadcast_shape = _np.broadcast_shapes(sample_shape, parameter_shape)
    except ValueError as exc:
        raise ValueError(
            "size and parameter arrays could not be broadcast together"
        ) from exc
    if broadcast_shape != sample_shape:
        raise ValueError("size and parameter arrays could not be broadcast together")
    return sample_shape


def _looks_like_shape(value):
    return _looks_like_integer_dimension(value) or (
        isinstance(value, tuple)
        and all(_looks_like_integer_dimension(dim) for dim in value)
    )


def _is_default_normal_scale(value):
    scale = _np.asarray(value)
    return scale.ndim == 0 and bool(scale == 1.0)


def _looks_like_shape_sequence(value):
    return isinstance(value, (list, tuple)) and all(
        _looks_like_integer_dimension(dim) for dim in value
    )


def _looks_like_scalar_randint_bound(value):
    value = _np.asarray(value)
    return value.shape == () and not _np.issubdtype(value.dtype, _np.bool_)


def set_state_return(has_state, state, res):
    if has_state:
        return state, res
    else:
        backend.jax_global_random_state = state
        return res


def _rand(state, size, *args, **kwargs):
    state, key = jax.random.split(state)
    return state, jax.random.uniform(key, _shape_from_size(size), *args, **kwargs)


def rand(*dims, size=None, **kwargs):
    state, has_state, kwargs = _get_state(**kwargs)
    state, res = _rand(state, _shape_from_rand_args(dims, size), **kwargs)
    return set_state_return(has_state, state, res)


def _contains_boolean_value(value):
    if isinstance(value, _BOOLEAN_TYPES):
        return True
    try:
        values = _np.asarray(value, dtype=object).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return False
    return any(isinstance(item, _BOOLEAN_TYPES) for item in values)


def _coerce_probability_values(value, name):
    if _contains_boolean_value(value):
        raise TypeError(f"{name} must be real numeric, not boolean")
    try:
        values = _np.asarray(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"{name} must be real numeric") from exc
    if values.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be real numeric")
    return values.astype(_np.float64, copy=False)


def _normalize_probability_values(values):
    if _np.any(values < 0.0) or _np.any(~_np.isfinite(values)):
        raise ValueError("probabilities do not sum to a positive value")
    scale = float(values.max()) if values.size else 0.0
    if scale <= 0.0:
        raise ValueError("probabilities do not sum to a positive value")
    scaled = values / scale
    total = float(scaled.sum())
    if not _np.isfinite(total) or total <= 0.0:
        raise ValueError("probabilities do not sum to a positive value")
    return _jnp.asarray(scaled / total)


def _validate_uniform_bound(bound, name):
    if _contains_boolean_value(bound):
        raise TypeError(f"{name} must be real numeric, not boolean")
    try:
        bound_array = _jnp.asarray(bound)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be real numeric") from exc
    if bound_array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be real numeric")
    if bool(_jnp.any(~_jnp.isfinite(bound_array))):
        raise ValueError("uniform bounds must be finite")
    return bound_array


def _validate_uniform_bounds(low, high):
    low = _validate_uniform_bound(low, "low")
    high = _validate_uniform_bound(high, "high")
    return low, high


def uniform(low=0.0, high=1.0, size=None, *args, **kwargs):
    low, high = _validate_uniform_bounds(low, high)
    shape = _bounded_sampler_shape(size, low, high)
    if bool(_jnp.any(low > high)):
        raise ValueError("Upper bound must be greater than or equal to lower bound")
    state, has_state, kwargs = _get_state(**kwargs)
    state, res = _rand(state, shape, *args, minval=low, maxval=high, **kwargs)
    return set_state_return(has_state, state, res)


def _validate_randint_bound(bound, name):
    if _contains_boolean_value(bound):
        raise TypeError(f"{name} must contain integer values")
    try:
        bound_array = _jnp.asarray(bound)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain integer values") from exc
    if bound_array.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integer values")
    return bound_array


def _validate_randint_bounds(low, high):
    try:
        low, high = _jnp.broadcast_arrays(low, high)
    except ValueError as exc:
        raise ValueError("low and high could not be broadcast together") from exc
    if bool(_jnp.any(high <= low)):
        raise ValueError("high must be greater than low")
    return low, high


def _randint(state, size, low, high, *args, **kwargs):
    state, key = jax.random.split(state)
    return state, jax.random.randint(
        key,
        _shape_from_size(size),
        low,
        high,
        *args,
        **kwargs,
    )


def randint(low=None, high=None, size=None, *args, **kwargs):
    """Draw integer samples using NumPy-compatible arguments.

    The preferred contract is ``randint(low, high=None, size=None, ...)``, which
    matches NumPy and the other PyRecEst backends. The older JAX-backend-only
    forms ``randint(shape, minval=..., maxval=...)``,
    ``randint(size=shape, minval=..., maxval=...)``, and
    ``randint(shape, minval, maxval)`` are still accepted for compatibility.
    """
    legacy_minval = kwargs.pop("minval", None)
    legacy_maxval = kwargs.pop("maxval", None)

    if legacy_minval is not None or legacy_maxval is not None:
        if legacy_minval is None or legacy_maxval is None:
            raise TypeError("Specify both 'minval' and 'maxval'.")
        if high is not None:
            raise TypeError(
                "Specify either NumPy-style 'low, high' or legacy "
                "'minval, maxval', not both."
            )
        if low is not None:
            if size is not None:
                raise TypeError(
                    "Specify the legacy output shape either as the first "
                    "positional argument or 'size=', not both."
                )
            size = low
        elif size is None:
            raise TypeError("randint() missing required argument 'size'")
        low = legacy_minval
        high = legacy_maxval
    elif (
        _looks_like_shape_sequence(low)
        and high is not None
        and size is not None
        and _looks_like_scalar_randint_bound(high)
        and _looks_like_scalar_randint_bound(size)
    ):
        # Legacy positional form: randint(shape, minval, maxval)
        size, low, high = low, high, size
    elif high is None:
        if low is None:
            raise TypeError("randint() missing required argument 'high'")
        high = low
        low = 0

    low = _validate_randint_bound(low, "low")
    high = _validate_randint_bound(high, "high")
    low, high = _validate_randint_bounds(low, high)
    shape = _bounded_sampler_shape(size, low, high)
    state, has_state, kwargs = _get_state(**kwargs)
    state, res = _randint(state, shape, low, high, *args, **kwargs)
    return set_state_return(has_state, state, res)


def _validate_normal_parameter(value, name):
    if _contains_boolean_value(value):
        raise TypeError(f"{name} must be real numeric, not boolean")
    try:
        value = _jnp.asarray(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"{name} must be real numeric") from exc
    if value.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be real numeric")
    if bool(_jnp.any(~_jnp.isfinite(value))):
        raise ValueError(f"{name} must be finite")
    return value


def _validate_normal_scale(scale):
    return _validate_normal_parameter(scale, "scale")


def _normal(state, loc=0.0, scale=1.0, size=None, *args, **kwargs):
    loc = _validate_normal_parameter(loc, "loc")
    scale = _validate_normal_scale(scale)
    if bool(_jnp.any(scale < 0)):
        raise ValueError("scale must be non-negative")
    sample_shape = _bounded_sampler_shape(size, loc, scale)
    state, key = jax.random.split(state)
    samples = jax.random.normal(key, sample_shape, *args, **kwargs)
    return state, loc + scale * samples


def normal(loc=0.0, scale=1.0, size=None, *args, **kwargs):
    """Draw samples using NumPy/PyTorch-compatible arguments.

    The legacy JAX-backend call form ``normal(size)`` is still accepted when the
    first positional argument looks like a shape and ``size`` is omitted.
    """
    if size is None and _looks_like_shape(loc) and _is_default_normal_scale(scale):
        size = loc
        loc = 0.0

    mean = kwargs.pop("mean", None)
    cov = kwargs.pop("cov", None)
    if mean is not None:
        loc = mean
    if cov is not None:
        scale = cov

    state, has_state, kwargs = _get_state(**kwargs)
    state, res = _normal(state, loc, scale, size, *args, **kwargs)
    return set_state_return(has_state, state, res)


def _integer_population_size(a):
    if isinstance(a, (int, _np.integer)) and not isinstance(a, (bool, _np.bool_)):
        return int(a)
    if a.ndim == 0 and _jnp.issubdtype(a.dtype, _jnp.integer):
        return int(a)
    return None


def _normalize_choice_axis(axis, ndim):
    if isinstance(axis, (bool, _np.bool_)):
        raise TypeError("axis must be an integer")
    try:
        axis = _operator_index(axis)
    except TypeError as exc:
        raise TypeError("axis must be an integer") from exc
    if axis < -ndim or axis >= ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    return axis % ndim


def _choice_bool(value, name):
    if isinstance(value, (bool, _np.bool_)):
        return bool(value)
    try:
        value_array = _jnp.asarray(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"{name} must be a boolean") from exc
    if value_array.shape == () and _jnp.issubdtype(value_array.dtype, _jnp.bool_):
        return bool(value_array.item())
    raise TypeError(f"{name} must be a boolean")


def _maybe_preserve_choice_order(indices, *, replace, p, shuffle):
    if replace or p is not None or shuffle:
        return indices

    index_array = _jnp.asarray(indices)
    if index_array.ndim == 0:
        return indices
    return _jnp.sort(index_array.reshape(-1)).reshape(index_array.shape)


def _choice_population_size(a, kwargs):
    population_size = _integer_population_size(a)
    if population_size is not None:
        return population_size

    if a.ndim == 0:
        raise ValueError(
            "a must be a positive integer or an array with at least one dimension"
        )

    axis = _normalize_choice_axis(kwargs.get("axis", 0), a.ndim)
    if "axis" in kwargs:
        kwargs["axis"] = axis
    return a.shape[axis]


def _empty_choice_result(a, shape, axis):
    if a.ndim == 0:
        return _jnp.empty(shape, dtype=a.dtype)
    return _jnp.empty(
        tuple(a.shape[:axis]) + tuple(shape) + tuple(a.shape[axis + 1 :]),
        dtype=a.dtype,
    )


def _validate_choice_probabilities(p, population_size):
    values = _coerce_probability_values(p, "p")
    if values.ndim != 1 or values.shape[0] != population_size:
        raise ValueError("p must be 1-dimensional with one entry per population item")
    return _normalize_probability_values(values)


def _choice(state, a, size=None, replace=True, p=None, shuffle=True, *args, **kwargs):
    if args:
        raise TypeError("choice() received unexpected positional arguments")
    state, key = jax.random.split(state)
    a = _jnp.asarray(a)
    shape = _shape_from_size(size)
    replace = _choice_bool(replace, "replace")
    shuffle = _choice_bool(shuffle, "shuffle")
    population_size = _choice_population_size(a, kwargs)
    axis = kwargs.get("axis", 0)
    if population_size <= 0:
        if _shape_has_no_samples(shape):
            return state, _empty_choice_result(a, shape, axis)
        raise ValueError("a must be a positive integer or a non-empty array")
    if p is not None:
        p = _validate_choice_probabilities(p, population_size)
    choice_kwargs = {name: value for name, value in kwargs.items() if name != "axis"}
    indices = jax.random.choice(
        key,
        population_size,
        shape=shape,
        replace=replace,
        p=p,
        **choice_kwargs,
    )
    indices = _maybe_preserve_choice_order(
        indices,
        replace=replace,
        p=p,
        shuffle=shuffle,
    )
    if a.ndim == 0:
        return state, indices
    return state, _jnp.take(a, indices, axis=axis)


def choice(a, size=None, replace=True, p=None, shuffle=True, *args, **kwargs):
    """Draw samples using a NumPy-like ``choice`` contract."""
    if "n" in kwargs:
        if size is not None:
            raise TypeError("Specify only one of 'size' or legacy 'n'.")
        size = kwargs.pop("n")
    state, has_state, kwargs = _get_state(**kwargs)
    state, res = _choice(state, a, size, replace, p, shuffle, *args, **kwargs)
    return set_state_return(has_state, state, res)


def _validate_multivariate_normal_mean(mean):
    mean = _validate_normal_parameter(mean, "mean")
    if mean.ndim != 1:
        raise ValueError("mean must be 1-dimensional")
    if mean.shape[0] == 0:
        raise ValueError("mean must contain at least one entry")
    return mean


def _validate_multivariate_normal_cov(cov, mean_dim):
    cov = _validate_normal_parameter(cov, "cov")
    if cov.ndim != 2:
        raise ValueError("cov must be a 2-dimensional square matrix")
    if cov.shape != (mean_dim, mean_dim):
        raise ValueError("cov must have shape (mean.size, mean.size)")
    cov_np = _np.asarray(cov)
    if not _np.allclose(cov_np, cov_np.T):
        raise ValueError("cov must be symmetric")
    cov_float = cov_np.astype(_np.result_type(cov_np.dtype, _np.float32), copy=False)
    eigenvalue_tolerance = 1e-8
    if _np.any(_np.linalg.eigvalsh(cov_float) < -eigenvalue_tolerance):
        raise ValueError("cov must be positive semidefinite")
    return cov


def _multivariate_normal(state, mean, cov, size=None, *args, **kwargs):
    state, key = jax.random.split(state)
    if "shape" in kwargs:
        if size is not None:
            raise TypeError("Specify only one of 'size' or 'shape'.")
        size = kwargs.pop("shape")
    shape = _shape_from_size(size)
    mean = _validate_multivariate_normal_mean(mean)
    cov = _validate_multivariate_normal_cov(cov, mean.shape[0])
    dtype = _jnp.result_type(mean, cov, _jnp.float32)
    mean = mean.astype(dtype)
    cov = cov.astype(dtype)
    return state, jax.random.multivariate_normal(key, mean, cov, shape, *args, **kwargs)


def multivariate_normal(mean, cov, size=None, *args, **kwargs):
    """Draw samples with NumPy-compatible ``multivariate_normal`` arguments."""
    state, has_state, kwargs = _get_state(**kwargs)
    state, res = _multivariate_normal(state, mean, cov, size, *args, **kwargs)
    return set_state_return(has_state, state, res)


def _validate_multinomial_pvals(pvals):
    values = _coerce_probability_values(pvals, "pvals")
    if values.ndim != 1:
        raise ValueError("pvals must be 1-dimensional")
    if values.shape[0] == 0:
        raise ValueError("pvals must contain at least one probability")
    return _normalize_probability_values(values)


def _multinomial(state, n, pvals, size=None):
    if not _looks_like_integer_dimension(n):
        raise TypeError("n must be a non-negative integer")
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")

    state, key = jax.random.split(state)
    sample_shape = _shape_from_size(size)
    pvals = _validate_multinomial_pvals(pvals)

    samples = jax.random.categorical(key, _jnp.log(pvals), shape=(*sample_shape, n))
    counts = _jnp.sum(
        jax.nn.one_hot(samples, pvals.shape[0], dtype=_jnp.int32), axis=-2
    )
    return state, counts


def multinomial(n, pvals, size=None, **kwargs):
    """Sample from a multinomial distribution using NumPy-compatible arguments."""
    state, has_state, kwargs = _get_state(**kwargs)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")
    state, res = _multinomial(state, n, pvals, size=size)
    return set_state_return(has_state, state, res)
