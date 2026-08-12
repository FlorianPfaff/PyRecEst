"""Compatibility shim for the JAX random backend module."""

from __future__ import annotations

import importlib.util as _importlib_util
import sys as _sys
import warnings as _warnings
from pathlib import Path as _Path

import numpy as _np

_LEGACY_MODULE_NAME = __name__ + "._legacy"
_LEGACY_PATH = _Path(__file__).resolve().parent.parent / "random.py"
_LEGACY_SPEC = _importlib_util.spec_from_file_location(
    _LEGACY_MODULE_NAME, _LEGACY_PATH
)
if _LEGACY_SPEC is None or _LEGACY_SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"Cannot load legacy JAX random backend from {_LEGACY_PATH}")
_LEGACY = _importlib_util.module_from_spec(_LEGACY_SPEC)
_sys.modules[_LEGACY_MODULE_NAME] = _LEGACY
_LEGACY_SPEC.loader.exec_module(_LEGACY)


def _validate_multivariate_normal_check_valid(check_valid):
    if not isinstance(check_valid, str) or check_valid not in {
        "warn",
        "raise",
        "ignore",
    }:
        raise ValueError("check_valid must be one of 'warn', 'raise', or 'ignore'")
    return check_valid


def _validate_multivariate_normal_tol(tol):
    message = "tol must be a finite non-negative scalar"
    if isinstance(tol, (str, bytes, bytearray)):
        raise ValueError(message)

    try:
        tol_array = _np.asarray(tol)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if tol_array.shape != () or _np.issubdtype(tol_array.dtype, _np.bool_):
        raise ValueError(message)
    scalar = tol_array.item()
    if isinstance(scalar, (bool, _np.bool_, str, bytes, bytearray)):
        raise ValueError(message)
    try:
        tol_value = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not _np.isfinite(tol_value) or tol_value < 0.0:
        raise ValueError(message)
    return tol_value


def _validate_and_classify_multivariate_normal_cov(cov, mean_dim, *, check_valid, tol):
    """Validate a covariance and identify numerically rank-deficient inputs."""

    cov = _LEGACY._validate_normal_parameter(cov, "cov")
    if cov.ndim != 2:
        raise ValueError("cov must be a 2-dimensional square matrix")
    if cov.shape != (mean_dim, mean_dim):
        raise ValueError("cov must have shape (mean.size, mean.size)")
    if not bool(_LEGACY._jnp.allclose(cov, cov.T)):
        raise ValueError("cov must be symmetric")

    cov_float = cov.astype(_LEGACY._jnp.result_type(cov, _LEGACY._jnp.float32))
    eigenvalues = _LEGACY._jnp.linalg.eigvalsh(cov_float)
    if bool(_LEGACY._jnp.any(eigenvalues < -tol)):
        message = "cov must be positive semidefinite"
        if check_valid == "raise":
            raise ValueError(message)
        if check_valid == "warn":
            _warnings.warn(message, RuntimeWarning, stacklevel=3)

    scale = _LEGACY._jnp.max(_LEGACY._jnp.abs(eigenvalues))
    rank_tolerance = _LEGACY._jnp.finfo(cov_float.dtype).eps * max(mean_dim, 1) * scale
    requires_svd = bool(_LEGACY._jnp.any(eigenvalues <= rank_tolerance))
    return cov, requires_svd


def multivariate_normal(mean, cov, size=None, *args, **kwargs):
    """Draw samples with NumPy-compatible validation keyword handling."""

    check_valid = kwargs.pop("check_valid", "warn")
    tol = kwargs.pop("tol", 1e-8)
    check_valid = _validate_multivariate_normal_check_valid(check_valid)
    tol = _validate_multivariate_normal_tol(tol)

    state, has_state, kwargs = _LEGACY._get_state(**kwargs)
    state, key = _LEGACY.jax.random.split(state)
    if "shape" in kwargs:
        if size is not None:
            raise TypeError("Specify only one of 'size' or 'shape'.")
        size = kwargs.pop("shape")
    shape = _LEGACY._shape_from_size(size)
    mean = _LEGACY._validate_multivariate_normal_mean(mean)
    cov, requires_svd = _validate_and_classify_multivariate_normal_cov(
        cov,
        mean.shape[0],
        check_valid=check_valid,
        tol=tol,
    )
    dtype = _LEGACY._jnp.result_type(mean, cov, _LEGACY._jnp.float32)
    mean = mean.astype(dtype)
    cov = cov.astype(dtype)
    if requires_svd and len(args) < 2 and "method" not in kwargs:
        kwargs["method"] = "svd"
    result = _LEGACY.jax.random.multivariate_normal(
        key, mean, cov, shape, *args, **kwargs
    )
    return _LEGACY.set_state_return(has_state, state, result)


__all__ = sorted(
    {
        name
        for name in dir(_LEGACY)
        if not (name.startswith("__") and name.endswith("__"))
    }
    | {"multivariate_normal"}
)


def __getattr__(name):
    return getattr(_LEGACY, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_LEGACY)))
