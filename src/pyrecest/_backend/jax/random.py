"""
Jax-based random backend.
Based on random.py by emilemathieu on
https://github.com/oxcsml/geomstats/blob/master/geomstats/_backend/jax/random.py
who says he was in inspired by https://github.com/wesselb/lab/blob/master/lab/jax/random.py
"""

import sys

import jax
import jax.numpy as _jnp
import numpy as _np

backend = sys.modules[__name__]


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


def _looks_like_integer_dimension(value):
    return isinstance(value, (int, _np.integer)) and not isinstance(
        value, (bool, _np.bool_)
    )


def _size_type_error():
    return TypeError("size must be None, an integer, or a sequence of integers")


def _integer_dimension(value):
    if not _looks_like_integer_dimension(value):
        raise _size_type_error()
    value = int(value)
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


def _looks_like_shape(value):
    return _looks_like_integer_dimension(value) or (
        isinstance(value, tuple)
        and all(_looks_like_integer_dimension(dim) for dim in value)
    )


def _looks_like_shape_sequence(value):
    return isinstance(value, (list, tuple)) and all(
        _looks_like_integer_dimension(dim) for dim in value
    )


def set_state_return(has_state, state, res):
    if has_state:
        return state, res
    else:
        backend.jax_global_random_state = state
        return res


def _rand(state, size, *args, **kwargs):
    state, key = jax.random.split(state)
    return state, jax.random.uniform(key, _shape_from_size(size), *args, **kwargs)


def rand(size=None, *args, **kwargs):
    state, has_state, kwargs = _get_state(**kwargs)
    state, res = _rand(state, size, *args, **kwargs)
    return set_state_return(has_state, state, res)


def uniform(low=0.0, high=1.0, size=None, *args, **kwargs):
    state, has_state, kwargs = _get_state(**kwargs)
    low = _jnp.asarray(low)
    high = _jnp.asarray(high)
    if bool(_jnp.any(low > high)):
        raise ValueError("Upper bound must be greater than or equal to lower bound")
    state, res = _rand(state, size, *args, minval=low, maxval=high, **kwargs)
    return set_state_return(has_state, state, res)


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
    elif _looks_like_shape_sequence(low) and high is not None and size is not None:
        # Legacy positional form: randint(shape, minval, maxval)
        size, low, high = low, high, size
    elif high is None:
        if low is None:
            raise TypeError("randint() missing required argument 'high'")
        high = low
        low = 0

    state, has_state, kwargs = _get_state(**kwargs)
    state, res = _randint(state, size, low, high, *args, **kwargs)
    return set_state_return(has_state, state, res)


def _normal(state, loc=0.0, scale=1.0, size=None, *args, **kwargs):
    state, key = jax.random.split(state)
    samples = jax.random.normal(key, _shape_from_size(size), *args, **kwargs)
    return state, loc + scale * samples


def normal(loc=0.0, scale=1.0, size=None, *args, **kwargs):
    """Draw samples using NumPy/PyTorch-compatible arguments.

    The legacy JAX-backend call form ``normal(size)`` is still accepted when the
    first positional argument looks like a shape and ``size`` is omitted.
    """
    if size is None and _looks_like_shape(loc) and scale == 1.0:
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


def _choice(state, a, size=None, replace=True, p=None, *args, **kwargs):
    state, key = jax.random.split(state)
    a = _jnp.asarray(a)
    if p is not None:
        p = _jnp.asarray(p, dtype=_jnp.float32)
        p = p / p.sum()
    res = jax.random.choice(
        key,
        a,
        *args,
        shape=_shape_from_size(size),
        replace=replace,
        p=p,
        **kwargs,
    )
    return state, res


def choice(a, size=None, replace=True, p=None, *args, **kwargs):
    """Draw samples using a NumPy-like ``choice`` contract."""
    if "n" in kwargs:
        if size is not None:
            raise TypeError("Specify only one of 'size' or legacy 'n'.")
        size = kwargs.pop("n")
    state, has_state, kwargs = _get_state(**kwargs)
    state, res = _choice(state, a, size, replace, p, *args, **kwargs)
    return set_state_return(has_state, state, res)


def _multivariate_normal(state, mean, cov, size=None, *args, **kwargs):
    state, key = jax.random.split(state)
    if "shape" in kwargs:
        if size is not None:
            raise TypeError("Specify only one of 'size' or 'shape'.")
        size = kwargs.pop("shape")
    shape = _shape_from_size(size)
    mean = _jnp.asarray(mean)
    cov = _jnp.asarray(cov)
    return state, jax.random.multivariate_normal(key, mean, cov, shape, *args, **kwargs)


def multivariate_normal(mean, cov, size=None, *args, **kwargs):
    """Draw samples with NumPy-compatible ``multivariate_normal`` arguments."""
    state, has_state, kwargs = _get_state(**kwargs)
    state, res = _multivariate_normal(state, mean, cov, size, *args, **kwargs)
    return set_state_return(has_state, state, res)


def _multinomial(state, n, pvals):
    state, key = jax.random.split(state)
    pvals = _jnp.asarray(pvals, dtype=_jnp.float32)
    pvals = pvals / pvals.sum()
    samples = jax.random.categorical(key, _jnp.log(pvals), shape=(n,))
    return state, _jnp.bincount(samples, minlength=len(pvals))


def multinomial(n, pvals, **kwargs):
    """Sample from a multinomial distribution using the JAX RNG state contract."""
    state, has_state, kwargs = _get_state(**kwargs)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")
    state, res = _multinomial(state, n, pvals)
    return set_state_return(has_state, state, res)
