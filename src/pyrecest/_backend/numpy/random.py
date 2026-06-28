"""Numpy based random backend."""

import numpy as _np
from numpy.random import default_rng as _default_rng
from numpy.random import (  # For PyRecEst
    get_state,
    seed,
    set_state,
)

from .._shared_numpy.random import choice, multivariate_normal, normal, rand, uniform

_BOOLEAN_TYPES = (bool, _np.bool_)


def _contains_boolean_value(value):
    if isinstance(value, _BOOLEAN_TYPES):
        return True
    try:
        values = _np.asarray(value, dtype=object).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return False
    return any(isinstance(item, _BOOLEAN_TYPES) for item in values)


def _validate_randint_bound(bound, name):
    if _contains_boolean_value(bound):
        raise TypeError(f"{name} must contain integer values")
    try:
        bound_array = _np.asarray(bound)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain integer values") from exc
    if bound_array.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integer values")


def randint(low, high=None, size=None, dtype=int):
    """Draw integer samples after rejecting non-integer bounds."""

    if high is None:
        _validate_randint_bound(low, "high")
        return _np.random.randint(low, high=None, size=size, dtype=dtype)

    _validate_randint_bound(low, "low")
    _validate_randint_bound(high, "high")
    return _np.random.randint(low, high=high, size=size, dtype=dtype)


def _validate_multinomial_sample_count(n):
    if _contains_boolean_value(n):
        raise TypeError("n must be a non-negative integer")
    try:
        n_array = _np.asarray(n)
    except (TypeError, ValueError) as exc:
        raise TypeError("n must be a non-negative integer") from exc
    if n_array.shape != () or n_array.dtype.kind not in "iu":
        raise TypeError("n must be a non-negative integer")
    count = int(n_array.item())
    if count < 0:
        raise ValueError("n must be non-negative")
    return count


def _validate_multinomial_pvals(pvals):
    if _contains_boolean_value(pvals):
        raise TypeError("pvals must be real numeric, not boolean")
    try:
        pvals_array = _np.asarray(pvals)
    except (TypeError, ValueError) as exc:
        raise TypeError("pvals must be real numeric") from exc
    if pvals_array.dtype.kind not in "iuf":
        raise TypeError("pvals must be real numeric")

    pvals_array = pvals_array.astype(float, copy=False)
    if pvals_array.ndim != 1:
        raise ValueError("pvals must be 1-dimensional")
    if pvals_array.size == 0:
        raise ValueError("pvals must contain at least one probability")

    pvals_sum = pvals_array.sum()
    if _np.any(pvals_array < 0) or not _np.isfinite(pvals_sum) or pvals_sum <= 0:
        raise ValueError("probabilities do not sum to a positive value")
    return pvals_array / pvals_sum


def _validate_multinomial_size(size):
    if size is not None and _contains_boolean_value(size):
        raise TypeError("size must be None, an integer, or a sequence of integers")
    return size


def multinomial(n, pvals, size=None):
    n = _validate_multinomial_sample_count(n)
    pvals_array = _validate_multinomial_pvals(pvals)
    size = _validate_multinomial_size(size)
    return _np.random.multinomial(n, pvals_array, size=size)
