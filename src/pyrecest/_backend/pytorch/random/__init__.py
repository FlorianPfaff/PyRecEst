"""Compatibility shim for the PyTorch random backend module."""

from __future__ import annotations

import importlib.util as _importlib_util
import sys as _sys
from pathlib import Path as _Path

import numpy as _np

_LEGACY_MODULE_NAME = __name__.rsplit(".", 1)[0] + "._random_legacy"
_LEGACY_PATH = _Path(__file__).resolve().parent.parent / "random.py"
_LEGACY_SPEC = _importlib_util.spec_from_file_location(
    _LEGACY_MODULE_NAME, _LEGACY_PATH
)
if _LEGACY_SPEC is None or _LEGACY_SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"Cannot load legacy PyTorch random backend from {_LEGACY_PATH}")
_LEGACY = _importlib_util.module_from_spec(_LEGACY_SPEC)
_sys.modules[_LEGACY_MODULE_NAME] = _LEGACY
_LEGACY_SPEC.loader.exec_module(_LEGACY)

_BOOLEAN_SCALAR_TYPES = (bool, _np.bool_)


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


def randint(low, high=None, size=None, *args, **kwargs):
    """Draw integer samples, rejecting boolean scalar bounds consistently."""

    _reject_boolean_randint_bound(low, "low" if high is not None else "high")
    if high is not None:
        _reject_boolean_randint_bound(high, "high")
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
