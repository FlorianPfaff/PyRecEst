"""Backend capability declarations used by documentation and tests.

The dynamic backend facade intentionally exposes the same attribute names for
all backends. Some attributes are native, bridged through NumPy/SciPy, partial, or explicitly
unsupported on a given backend. Keeping those declarations in one lightweight
module gives tests and documentation a single source of truth.
"""

from __future__ import annotations

from operator import index as _operator_index
from typing import Any, Final, cast

BACKEND_NAMES: Final = ("numpy", "pytorch", "jax")

BACKEND_CAPABILITIES: Final = {
    "numpy": {
        "unsupported": {},
        "partial": {},
    },
    "pytorch": {
        "unsupported": {
            "": ("searchsorted",),
        },
        "bridged": {
            "linalg": {
                "sqrtm": "SciPy bridge; not differentiable through the bridge.",
                "fractional_matrix_power": "SciPy bridge; not differentiable through the bridge.",
                "polar": "SciPy bridge; not differentiable through the bridge.",
                "quadratic_assignment": "SciPy bridge; returns Python indices.",
            },
        },
        "partial": {
            "linalg": {
                "solve_sylvester": "Uses native fast paths and falls back to SciPy.",
            },
        },
    },
    "jax": {
        "unsupported": {
            "": (
                "convert_to_wider_dtype",
                "get_default_dtype",
                "get_default_cdtype",
            ),
            "autodiff": (
                "hessian",
                "hessian_vec",
                "jacobian_vec",
                "jacobian_and_hessian",
                "value_jacobian_and_hessian",
                "value_and_jacobian",
            ),
            "linalg": (
                "fractional_matrix_power",
                "logm",
                "quadratic_assignment",
            ),
        },
        "bridged": {},
        "partial": {
            "random": {
                "module": "Global PRNG state is provided for facade compatibility; explicit state passing is preferred for JAX workflows.",
            },
        },
    },
}

API_BACKEND_CAPABILITIES: Final = {
    "KalmanFilter": {
        "numpy": "supported",
        "pytorch": "supported",
        "jax": "supported",
        "notes": "Linear Gaussian operations are part of the portable baseline.",
    },
    "UnscentedKalmanFilter": {
        "numpy": "supported",
        "pytorch": "partial",
        "jax": "partial",
        "notes": "Portable for backend-compatible model functions; advanced paths may still bridge through NumPy/SciPy.",
    },
    "EuclideanParticleFilter": {
        "numpy": "supported",
        "pytorch": "partial",
        "jax": "partial",
        "notes": "Particle operations are portable where sampling and resampling helpers preserve backend semantics.",
    },
    "DistributionConversion": {
        "numpy": "supported",
        "pytorch": "partial",
        "jax": "partial",
        "notes": "Euclidean particle/Gaussian conversions are portable; grid, Fourier, and manifold routes are route-specific.",
    },
    "UKFOnManifolds": {
        "numpy": "supported",
        "pytorch": "partial",
        "jax": "unsupported",
        "notes": "The current implementation documents explicit JAX exclusions for predict/update.",
    },
    "SphericalHarmonicsEOTTracker": {
        "numpy": "supported",
        "pytorch": "unsupported",
        "jax": "unsupported",
        "notes": "Depends on spherical harmonics and SciPy-adjacent functionality.",
    },
    "GaussianDistribution": {
        "numpy": "supported",
        "pytorch": "supported",
        "jax": "supported",
        "notes": "Basic construction, moment access, and portable operations should remain backend portable.",
    },
    "LinearDiracDistribution": {
        "numpy": "supported",
        "pytorch": "supported",
        "jax": "supported",
        "notes": "Used by representation conversion and particle-style workflows.",
    },
    "MultiBernoulliTracker": {
        "numpy": "supported",
        "pytorch": "partial",
        "jax": "unsupported",
        "notes": "Tracking workflows rely on assignment and measurement-set utilities that are currently NumPy-oriented.",
    },
    "PointSetRegistration": {
        "numpy": "supported",
        "pytorch": "partial",
        "jax": "unsupported",
        "notes": "Registration utilities may copy through NumPy/SciPy and should not be assumed differentiable.",
    },
    "EvaluationUtilities": {
        "numpy": "supported",
        "pytorch": "bridged",
        "jax": "bridged",
        "notes": "Some plotting, assignment, and summary operations remain NumPy/SciPy oriented and may not preserve device or gradient semantics.",
    },
    "BackendFacade": {
        "numpy": "supported",
        "pytorch": "partial",
        "jax": "partial",
        "notes": "Facade names are importable across backends, but some functions are bridged or explicitly unsupported.",
    },
    "DiscreteStateUtilities": {
        "numpy": "supported",
        "pytorch": "bridged",
        "jax": "bridged",
        "notes": "Finite-state HMM and IMM utilities operate on NumPy arrays and SciPy sparse matrices; non-NumPy inputs are coerced.",
    },
}

BACKEND_SUPPORT_LEVELS: Final = ("supported", "bridged", "partial", "unsupported")
REQUIRED_BACKENDS: Final = ("numpy", "pytorch", "jax")
_OPTIONAL_API_CAPABILITY_KEYS: Final = frozenset({"notes"})
_ALLOWED_API_CAPABILITY_KEYS: Final = frozenset(
    (*REQUIRED_BACKENDS, *_OPTIONAL_API_CAPABILITY_KEYS)
)
_PYTORCH_ARGSORT_DEFAULT_AXIS: Final = object()
_JAX_ARGSORT_DEFAULT_AXIS: Final = object()


def _patch_jax_backend_contracts() -> None:
    try:
        from pyrecest.backend_support._jax_random_empty_contract import (  # pylint: disable=import-outside-toplevel
            patch_jax_randint_empty_size_contract,
        )
    except ModuleNotFoundError:  # pragma: no cover - backend support may be unavailable
        return
    patch_jax_randint_empty_size_contract()


def _patch_random_backend_contracts() -> None:
    try:
        from pyrecest.backend_support._random_uniform_empty_contract import (  # pylint: disable=import-outside-toplevel
            patch_random_uniform_empty_bounds_contract,
        )
    except ModuleNotFoundError:  # pragma: no cover - backend support may be unavailable
        return
    patch_random_uniform_empty_bounds_contract()


def _normalize_pytorch_argsort_axis(axis, torch_module) -> int:
    """Return one NumPy-style argsort axis without bool-as-int coercion."""
    if isinstance(axis, bool):
        raise TypeError("an integer is required for the axis")
    if torch_module.is_tensor(axis):
        if axis.ndim != 0 or axis.dtype == torch_module.bool:
            raise TypeError("an integer is required for the axis")
    try:
        return _operator_index(axis)
    except TypeError as exc:
        raise TypeError("an integer is required for the axis") from exc


def _resolve_pytorch_argsort_axis(axis, dim, torch_module) -> int | None:
    """Resolve NumPy ``axis`` and PyTorch ``dim`` aliases for argsort."""
    axis_was_omitted = axis is _PYTORCH_ARGSORT_DEFAULT_AXIS
    if axis_was_omitted:
        axis = -1
    if dim is not None:
        dim_value = _normalize_pytorch_argsort_axis(dim, torch_module)
        if not axis_was_omitted and axis is not None:
            axis_value = _normalize_pytorch_argsort_axis(axis, torch_module)
            if axis_value != dim_value:
                raise TypeError("argsort() got both 'axis' and 'dim'")
        return dim_value
    if axis is None:
        return None
    return _normalize_pytorch_argsort_axis(axis, torch_module)


def _raise_pytorch_argsort_kind_stable_conflict() -> None:
    """Raise NumPy's error for simultaneous ``kind`` and ``stable`` arguments."""
    raise ValueError(
        "`kind` and `stable` parameters can't be provided at the same time. "
        "Use only one of them."
    )


def _normalize_jax_argsort_axis(axis) -> int:
    """Return one NumPy-style JAX argsort axis without bool-as-int coercion."""
    if isinstance(axis, bool) or type(axis).__name__ == "bool_":
        raise TypeError("an integer is required for the axis")
    try:
        return _operator_index(axis)
    except TypeError as exc:
        raise TypeError("an integer is required for the axis") from exc


def _resolve_jax_argsort_axis(axis, dim) -> int | None:
    """Resolve NumPy ``axis`` and optional PyTorch-style ``dim`` aliases."""
    axis_was_omitted = axis is _JAX_ARGSORT_DEFAULT_AXIS
    if axis_was_omitted:
        axis = -1
    if dim is not None:
        dim_value = _normalize_jax_argsort_axis(dim)
        if not axis_was_omitted and axis is not None:
            axis_value = _normalize_jax_argsort_axis(axis)
            if axis_value != dim_value:
                raise TypeError("argsort() got both 'axis' and 'dim'")
        return dim_value
    if axis is None:
        return None
    return _normalize_jax_argsort_axis(axis)


def _patch_jax_argsort_contracts() -> None:
    try:
        import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.jax as raw_jax  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - JAX backend may be unavailable
        return

    original_argsort = getattr(raw_jax, "argsort", None)
    if original_argsort is None:
        return
    if getattr(original_argsort, "_pyrecest_arraylike_contract", False):
        if getattr(backend, "__backend_name__", None) == "jax":
            backend.argsort = original_argsort
        return

    def argsort(
        a,
        axis=_JAX_ARGSORT_DEFAULT_AXIS,
        kind=None,
        order=None,
        *,
        stable=True,
        descending=False,
        dim=None,
    ):
        axis_value = _resolve_jax_argsort_axis(axis, dim)
        return jnp.argsort(
            jnp.asarray(a),
            axis=axis_value,
            kind=kind,
            order=order,
            stable=stable,
            descending=descending,
        )

    argsort.__name__ = getattr(original_argsort, "__name__", "argsort")
    argsort.__doc__ = getattr(original_argsort, "__doc__", None)
    argsort._pyrecest_arraylike_contract = True
    argsort._pyrecest_numpy_contract = True
    raw_jax.argsort = argsort
    if getattr(backend, "__backend_name__", None) == "jax":
        backend.argsort = argsort


def _patch_pytorch_argsort_contracts() -> None:
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    original_argsort = getattr(raw_pytorch, "argsort", None)
    if original_argsort is None:
        return
    if getattr(original_argsort, "_pyrecest_numpy_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.argsort = original_argsort
        return

    def argsort(
        a,
        axis=_PYTORCH_ARGSORT_DEFAULT_AXIS,
        kind=None,
        order=None,
        *,
        stable=None,
        dim=None,
        descending=False,
    ):
        axis_value = _resolve_pytorch_argsort_axis(axis, dim, torch)
        if order is not None:
            raise ValueError("order is not supported by the PyTorch backend")
        if kind is not None and stable is not None:
            _raise_pytorch_argsort_kind_stable_conflict()
        if kind is not None:
            if kind in {"stable", "mergesort"}:
                stable = True
            elif kind in {"quicksort", "heapsort"}:
                stable = False
            else:
                raise ValueError(
                    "sort kind must be one of 'quicksort', 'heapsort', 'stable', or 'mergesort'"
                )

        values = raw_pytorch.array(a)
        if axis_value is None:
            values = values.reshape(-1)
            axis_value = 0
        return torch.argsort(
            values,
            dim=axis_value,
            descending=descending,
            stable=bool(stable) if stable is not None else False,
        )

    argsort.__name__ = getattr(original_argsort, "__name__", "argsort")
    argsort.__doc__ = getattr(original_argsort, "__doc__", None)
    argsort._pyrecest_arraylike_contract = True
    argsort._pyrecest_numpy_contract = True
    raw_pytorch.argsort = argsort
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.argsort = argsort


_patch_jax_backend_contracts()
_patch_random_backend_contracts()
_patch_jax_argsort_contracts()
_patch_pytorch_argsort_contracts()


def get_unsupported_functions(
    backend_name: str, module_name: str = ""
) -> tuple[str, ...]:
    """Return unsupported facade functions for a backend module."""
    backend = cast(dict[str, Any], BACKEND_CAPABILITIES.get(backend_name, {}))
    unsupported = cast(dict[str, tuple[str, ...]], backend.get("unsupported", {}))
    return tuple(unsupported.get(module_name, ()))


def get_partial_capabilities(
    backend_name: str, module_name: str = ""
) -> dict[str, str]:
    """Return partial-support notes for a backend module."""
    backend = cast(dict[str, Any], BACKEND_CAPABILITIES.get(backend_name, {}))
    partial = cast(dict[str, dict[str, str]], backend.get("partial", {}))
    return dict(partial.get(module_name, {}))


def get_bridged_capabilities(
    backend_name: str, module_name: str = ""
) -> dict[str, str]:
    """Return operations that work by crossing into another numerical stack."""
    backend = cast(dict[str, Any], BACKEND_CAPABILITIES.get(backend_name, {}))
    bridged = cast(dict[str, dict[str, str]], backend.get("bridged", {}))
    return dict(bridged.get(module_name, {}))


def get_api_backend_support(api_name: str) -> dict[str, str]:
    """Return backend support metadata for a public API name."""
    return dict(API_BACKEND_CAPABILITIES.get(api_name, {}))


def iter_api_backend_capabilities() -> tuple[tuple[str, dict[str, str]], ...]:
    """Return public API backend support rows in a stable order."""
    return tuple(
        (api_name, dict(row))
        for api_name, row in sorted(API_BACKEND_CAPABILITIES.items())
    )


def validate_api_backend_capabilities() -> tuple[str, ...]:
    """Return human-readable validation errors for API capability metadata."""
    errors: list[str] = []
    for api_name, row in iter_api_backend_capabilities():
        if not api_name:
            errors.append("Capability row has an empty API name.")

        unknown_keys = sorted(set(row) - _ALLOWED_API_CAPABILITY_KEYS)
        if unknown_keys:
            errors.append(
                f"{api_name}: unknown capability entries for {', '.join(unknown_keys)}."
            )

        missing_backends = [
            backend for backend in REQUIRED_BACKENDS if backend not in row
        ]
        if missing_backends:
            errors.append(
                f"{api_name}: missing backend support entries for {', '.join(missing_backends)}."
            )

        for backend_name in REQUIRED_BACKENDS:
            support_level = row.get(backend_name)
            if support_level not in BACKEND_SUPPORT_LEVELS:
                errors.append(
                    f"{api_name}: unsupported support level {support_level!r} for {backend_name}."
                )

        if not row.get("notes"):
            errors.append(f"{api_name}: missing explanatory notes.")

    return tuple(errors)
