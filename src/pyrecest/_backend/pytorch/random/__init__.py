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
_PROBABILITY_SUM_ERROR = "probabilities do not sum to a positive value"
_UNIFORM_RANGE_ERROR = "high - low range exceeds valid bounds"


def _probability_accumulation_dtype(probabilities):
    torch = _LEGACY._torch
    if probabilities.dtype.is_floating_point:
        return torch.promote_types(probabilities.dtype, torch.get_default_dtype())
    return torch.get_default_dtype()


def _normalize_nonnegative_probabilities(probabilities):
    torch = _LEGACY._torch
    probabilities = probabilities.to(
        dtype=_probability_accumulation_dtype(probabilities)
    )
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


def _as_probability_tensor(values, device):
    """Convert array-like probabilities without narrowing Python floats."""

    torch = _LEGACY._torch
    if torch.is_tensor(values):
        return values.to(device=device)
    return torch.as_tensor(_np.asarray(values), device=device)


def _validate_choice_probabilities(p, population_size, device):
    if _LEGACY._contains_boolean_value(p):
        raise TypeError("p must be real numeric, not boolean")
    try:
        p = _as_probability_tensor(p, device)
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
        pvals = _as_probability_tensor(pvals, device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError("pvals must be real numeric") from exc
    if not _LEGACY._is_real_numeric_dtype(pvals.dtype):
        raise TypeError("pvals must be real numeric")
    if pvals.numel() == 0:
        return pvals.to(dtype=_probability_accumulation_dtype(pvals))
    return _normalize_nonnegative_probabilities(pvals)


_LEGACY._validate_choice_probabilities = _validate_choice_probabilities
_LEGACY._validate_multinomial_pvals = _validate_multinomial_pvals


def _promoted_multivariate_normal_dtype(*values):
    """Promote precision across all tensor-valued distribution parameters."""

    torch = _LEGACY._torch
    promoted_dtype = None
    for value in values:
        if not torch.is_tensor(value):
            continue
        value_dtype = value.dtype
        if value_dtype.is_complex:
            value_dtype = _LEGACY._COMPLEX_TO_FLOAT_DTYPE[value_dtype]
        elif not value_dtype.is_floating_point:
            continue
        promoted_dtype = (
            value_dtype
            if promoted_dtype is None
            else torch.promote_types(promoted_dtype, value_dtype)
        )
    return promoted_dtype or torch.get_default_dtype()


_LEGACY._floating_distribution_dtype = _promoted_multivariate_normal_dtype


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
    try:
        tol_array = _np.asarray(tol)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if (
        tol_array.shape != ()
        or tol_array.dtype.kind in "Mm"
        or _np.issubdtype(tol_array.dtype, _np.bool_)
    ):
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
        (value.device for value in values if _LEGACY._torch.is_tensor(value)),
        None,
    )


_LEGACY_ARRAY_RANDINT = _LEGACY._randint_array


def _sample_array_randint_exactly(low, high, dtype, generator):
    """Sample array-valued bounds without passing through floating-point values."""

    torch = _LEGACY._torch
    flat_low = low.reshape(-1).to(dtype=torch.int64)
    flat_high = high.reshape(-1).to(dtype=torch.int64)
    flat_result = torch.empty(flat_low.shape, dtype=dtype, device=low.device)
    if flat_low.numel() == 0:
        return flat_result.reshape(low.shape)

    bounds = torch.stack((flat_low, flat_high), dim=1)
    unique_bounds, inverse = torch.unique(bounds, dim=0, return_inverse=True)
    order = torch.argsort(inverse)
    counts = torch.bincount(inverse, minlength=unique_bounds.shape[0]).tolist()

    offset = 0
    for bound_pair, count in zip(unique_bounds, counts):
        positions = order[offset : offset + count]
        flat_result[positions] = torch.randint(
            int(bound_pair[0].item()),
            int(bound_pair[1].item()),
            (count,),
            dtype=dtype,
            device=low.device,
            generator=generator,
        )
        offset += count

    return flat_result.reshape(low.shape)


def _randint_array_with_wide_arithmetic(low, high, size, *args, **kwargs):
    """Sample array bounds exactly while retaining wide integer arithmetic."""

    if args:
        return _LEGACY_ARRAY_RANDINT(low, high, size, *args, **kwargs)

    torch = _LEGACY._torch
    requested_dtype = _LEGACY._normalize_randint_dtype(kwargs.get("dtype"))
    device = _LEGACY._randint_device(low, high, device=kwargs.get("device"))
    low = torch.as_tensor(low, device=device)
    high = torch.as_tensor(high, device=device)
    _LEGACY._validate_randint_array_bound("low", low)
    _LEGACY._validate_randint_array_bound("high", high)
    sample_shape = _LEGACY._randint_array_size(size, low, high)
    try:
        low = torch.broadcast_to(low, sample_shape)
        high = torch.broadcast_to(high, sample_shape)
    except RuntimeError as exc:
        raise ValueError("size, low, and high could not be broadcast together") from exc
    if bool(torch.any(high <= low)):
        raise ValueError("high must be greater than low")
    _LEGACY._validate_randint_array_dtype_bounds(low, high, requested_dtype)

    sampling_kwargs = dict(kwargs)
    sampling_kwargs.pop("dtype", None)
    sampling_kwargs.pop("device", None)
    generator = sampling_kwargs.pop("generator", None)
    out = sampling_kwargs.pop("out", None)
    if sampling_kwargs:
        unexpected = ", ".join(sorted(sampling_kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

    result = _sample_array_randint_exactly(low, high, requested_dtype, generator)
    if out is not None:
        out.copy_(result)
        return out
    return result


_LEGACY._randint_array = _randint_array_with_wide_arithmetic


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


def uniform(low=0.0, high=1.0, size=None, dtype=None):
    """Draw uniform samples while rejecting non-representable finite ranges."""

    torch = _LEGACY._torch
    dtype = _LEGACY._normalize_random_dtype(dtype, default=None)
    device = _LEGACY._tensor_device(low, high)
    low = _LEGACY._validate_uniform_bound(low, "low", dtype=dtype, device=device)
    high = _LEGACY._validate_uniform_bound(high, "high", dtype=dtype, device=device)
    size = _LEGACY._uniform_size(size, low, high)
    _LEGACY._validate_uniform_bounds(low, high)

    arithmetic_dtype = dtype or torch.promote_types(
        torch.result_type(low, high), torch.get_default_dtype()
    )
    low = low.to(dtype=arithmetic_dtype)
    high = high.to(dtype=arithmetic_dtype)
    span = high - low
    if bool(torch.any(~torch.isfinite(span))):
        raise OverflowError(_UNIFORM_RANGE_ERROR)
    return span * torch.rand(size, dtype=arithmetic_dtype, device=device) + low


def _singular_multivariate_normal_factor(mean, cov, tol):
    """Return a square-root factor for a valid singular covariance."""

    torch = _LEGACY._torch
    device = _LEGACY._tensor_device(mean, cov)
    dtype = _LEGACY._floating_distribution_dtype(mean, cov)
    try:
        mean = _LEGACY._validate_multivariate_normal_parameter(
            mean, "mean", dtype=dtype, device=device
        )
        cov = _LEGACY._validate_multivariate_normal_parameter(
            cov, "cov", dtype=mean.dtype, device=mean.device
        )
    except (TypeError, ValueError, RuntimeError):
        return None

    if mean.ndim != 1 or cov.ndim != 2:
        return None
    if cov.shape != (mean.shape[0], mean.shape[0]) or mean.shape[0] == 0:
        return None
    if not bool(torch.allclose(cov, cov.T, rtol=0.0, atol=tol)):
        return None

    symmetric_cov = 0.5 * (cov + cov.T)
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(symmetric_cov)
    except RuntimeError:
        return None
    if bool(torch.any(eigenvalues < -tol)):
        return None

    scale = torch.max(torch.abs(eigenvalues))
    rank_tolerance = torch.finfo(cov.dtype).eps * max(mean.shape[0], 1) * scale
    if bool(torch.all(eigenvalues > rank_tolerance)):
        return None

    factor = eigenvectors * torch.sqrt(torch.clamp(eigenvalues, min=0.0)).unsqueeze(0)
    return mean, factor


def _sample_singular_multivariate_normal(mean, factor, size):
    """Sample a singular Gaussian through its eigendecomposition factor."""

    torch = _LEGACY._torch
    sample_shape = _LEGACY._normal_sample_size(size)
    standard_normal = torch.randn(
        (*sample_shape, mean.shape[0]),
        dtype=mean.dtype,
        device=mean.device,
    )
    return mean + torch.matmul(standard_normal, factor.T)


def multivariate_normal(mean, cov, size=None, *args, **kwargs):
    """Draw samples with NumPy-compatible validation keyword handling."""

    check_valid = kwargs.pop("check_valid", "warn")
    tol = kwargs.pop("tol", 1e-8)
    _validate_multivariate_normal_check_valid(check_valid)
    tol = _validate_multivariate_normal_tol(tol)

    try:
        return _LEGACY.multivariate_normal(mean, cov, size=size, *args, **kwargs)
    except ValueError:
        if args or kwargs:
            raise
        singular_parameters = _singular_multivariate_normal_factor(mean, cov, tol)
        if singular_parameters is None:
            raise
        singular_mean, factor = singular_parameters
        return _sample_singular_multivariate_normal(singular_mean, factor, size)


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
