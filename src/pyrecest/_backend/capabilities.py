"""Backend capability declarations used by documentation and tests.

The dynamic backend facade intentionally exposes the same attribute names for
all backends. Some attributes are native, bridged through NumPy/SciPy, partial, or explicitly
unsupported on a given backend. Keeping those declarations in one lightweight
module gives tests and documentation a single source of truth.
"""

from __future__ import annotations

from typing import Final

BACKEND_NAMES: Final = ("numpy", "pytorch", "jax")

BACKEND_CAPABILITIES: Final = {
    "numpy": {
        "unsupported": {},
        "partial": {},
    },
    "pytorch": {
        "unsupported": {
            "": ("searchsorted",),
            "signal": ("fftconvolve",),
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
            "random": {
                "choice": "Weighted sampling without replacement is not supported.",
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
                "is_single_matrix_pd",
                "logm",
                "quadratic_assignment",
                "solve_sylvester",
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


def get_unsupported_functions(
    backend_name: str, module_name: str = ""
) -> tuple[str, ...]:
    """Return unsupported facade functions for a backend module."""
    backend = BACKEND_CAPABILITIES.get(backend_name, {})
    unsupported = backend.get("unsupported", {})
    return tuple(unsupported.get(module_name, ()))


def get_partial_capabilities(
    backend_name: str, module_name: str = ""
) -> dict[str, str]:
    """Return partial-support notes for a backend module."""
    backend = BACKEND_CAPABILITIES.get(backend_name, {})
    partial = backend.get("partial", {})
    return dict(partial.get(module_name, {}))


def get_bridged_capabilities(
    backend_name: str, module_name: str = ""
) -> dict[str, str]:
    """Return operations that work by crossing into another numerical stack."""
    backend = BACKEND_CAPABILITIES.get(backend_name, {})
    bridged = backend.get("bridged", {})
    return dict(bridged.get(module_name, {}))


def get_api_backend_support(api_name: str) -> dict[str, str]:
    """Return backend support metadata for a public API name."""
    return dict(API_BACKEND_CAPABILITIES.get(api_name, {}))


def iter_api_backend_capabilities() -> tuple[tuple[str, dict[str, str]], ...]:
    """Return public API backend support rows in a stable order."""
    return tuple(sorted(API_BACKEND_CAPABILITIES.items()))


def validate_api_backend_capabilities() -> tuple[str, ...]:
    """Return human-readable validation errors for API capability metadata."""
    errors: list[str] = []
    for api_name, row in iter_api_backend_capabilities():
        if not api_name:
            errors.append("Capability row has an empty API name.")

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
