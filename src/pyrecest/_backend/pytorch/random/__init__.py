"""Compatibility shim for the PyTorch random backend module."""

from __future__ import annotations

import importlib.util as _importlib_util
import sys as _sys
from pathlib import Path as _Path

import numpy as _np

_LEGACY_MODULE_NAME = __name__.rsplit(".", 1)[0] + "._random_legacy"
_LEGACY_PATH = _Path(__file__).resolve().parent.parent / "random.py"
_LEGACY_SPEC = _importlib_util.spec_from_file_location(_LEGACY_MODULE_NAME, _LEGACY_PATH)
if _LEGACY_SPEC is None or _LEGACY_SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"Cannot load legacy PyTorch random backend from {_LEGACY_PATH}")
_LEGACY = _importlib_util.module_from_spec(_LEGACY_SPEC)
_sys.modules[_LEGACY_MODULE_NAME] = _LEGACY
_LEGACY_SPEC.loader.exec_module(_LEGACY)

_BOOLEAN_SCALAR_TYPES = (bool, _np.bool_)
_PROBABILITY_SUM_ERROR = "probabilities do not sum to a positive value"


def _probability_accumulation_dtype(probabilities):
    torch = _LEGACY._torch
    if probabilities.dtype.is_floating_point:
        return torch.promote_types(probabilities.dtype, torch.get_default_dtype())
    return torch.get_default_dtype()


def _normalize_nonnegative_probabilities(probabilities):
    torch = _LEGACY._torch
    probabilities = probabilities.to(dtype=_probability_accumulation_dtype(probabilities))
    if bool(torch.any(probabilities < 0)):
        raise ValueError(_PROBABILITY_SUM_ERROR)
    scale = probabilities.max()
    if not bool(torch.isfinite(scale)) or bool(scale <= 0):
        raise ValueError(_PROBABILITY_SUM_ERROR)
    scaled = probabilities / scale
    total = scaled.sum()
    if not bool(torch.isfinite(total)) or bool(total <= 0):
        raise ValueError(_PROBABILITY_SUM_ERROR)
    return scaled / total


def _validate_choice_probabilities(p, population_size, device):
    if _LEGACY._contains_boolean_value(p):
        raise TypeError("p must be real numeric, not boolean")
    try:
        p = _LEGACY._torch.as_tensor(p, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError("p must be real numeric") from exc
    if not _LEGACY._is_real_numeric_dtype(p.dtype):
        raise TypeError("p must be real numeric")
    if p.ndim != 1 or p.shape[0] != population_size:
        raise ValueError("p must be 1-dimensional with one entry per population item")
    return _normalize_nonnegative_probabilities(p)


def _validate_multinomial_pvals(pvals, device):
    if _LEGACY._contains_boolean_value(pvals):
        raise TypeError("pvals must be real numeric, not boolean")
    try:
        pvals = _LEGACY._torch.as_tensor(pvals, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError("pvals must be real numeric") from exc
    if not _LEGACY._is_real_numeric_dtype(pvals.dtype):
        raise TypeError("pvals must be real numeric")
    if pvals.numel() == 0:
        return pvals.to(dtype=_probability_accumulation_dtype(pvals))
    return _normalize_nonnegative_probabilities(pvals)


_LEGACY._validate_choice_probabilities = _validate_choice_probabilities
_LEGACY._validate_multinomial_pvals = _validate_multinomial_pvals


def _validate_multivariate_normal_check_valid(check_valid):
    if check_valid not in {"warn", "raise", "ignore"}:
        raise ValueError("check_valid must be one of 'warn', 'raise', or 'ignore'")
    return check_valid


def _validate_multivariate_normal_tol(tol):
    message = "tol must be a finite non-negative scalar"
    try:
        tol_array = _np.asarray(tol)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if tol_array.shape != () or _np.issubdtype(tol_array.dtype, _np.bool_):
        raise ValueError(message)
    scalar = tol_array.item()
    if isinstance(scalar, (bool, _np.bool_, str, bytes, _np.str_, _np.bytes_)):
        raise ValueError(message)
    try:
        tol_value = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not _np.isfinite(tol_value) or tol_value < 0.0:
        raise ValueError(message)
    return tol_value


def _reject_boolean_randint_bound(value, name):
    if isinstance(value, _BOOLEAN_SCALAR_TYPES):
        raise TypeError(f"{name} must contain integer values")


def _normalize_scalar_integer_randint_bound(value):
    """Convert integer-like scalar arrays/tensors to exact Python integers."""
    scalar = _LEGACY._scalar_integer_dimension(value)
    return value if scalar is None else scalar


def _first_randint_bound_device(*values):
    """Return the first tensor-bound device so scalar normalization preserves it."""
    return next(
        (
            value.device
            for value in values
            if _LEGACY._torch.is_tensor(value)
        ),
        None,
    )


def randint(low, high=None, size=None, *args, **kwargs):
    """Draw integer samples with exact scalar-bound handling."""

    _reject_boolean_randint_bound(low, "low" if high is not None else "high")
    if high is not None:
        _reject_boolean_randint_bound(high, "high")

    bound_device = _first_randint_bound_device(low, high)
    low = _normalize_scalar_integer_randint_bound(low)
    if high is not None:
        high = _normalize_scalar_integer_randint_bound(high)

    if bound_device is not None and "device" not in kwargs:
        kwargs = dict(kwargs)
        kwargs["device"] = bound_device

    return _LEGACY.randint(low, high, size, *args, **kwargs)


def multivariate_normal(mean, cov, size=None, *args, **kwargs):
    """Draw samples with NumPy-compatible validation keyword handling."""

    check_valid = kwargs.pop("check_valid", "warn")
    tol = kwargs.pop("tol", 1e-8)
    _validate_multivariate_normal_check_valid(check_valid)
    _validate_multivariate_normal_tol(tol)
    return _LEGACY.multivariate_normal(mean, cov, size=size, *args, **kwargs)


__all__ = sorted(
    {
        name
        for name in dir(_LEGACY)
        if not (name.startswith("__") and name.endswith("__"))
    }
    | {"multivariate_normal", "randint"}
)


def __getattr__(name):
    return getattr(_LEGACY, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_LEGACY)))
