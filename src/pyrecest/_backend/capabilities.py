"""Backend capability declarations used by documentation and tests.

The dynamic backend facade intentionally exposes the same attribute names for
all backends. Some attributes are partial or explicitly unsupported on a given
backend. Keeping those declarations in one lightweight module gives tests and
documentation a single source of truth.
"""

from __future__ import annotations

from typing import Final

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
        "partial": {
            "linalg": {
                "sqrtm": "SciPy bridge; not differentiable through the bridge.",
                "fractional_matrix_power": "SciPy bridge; not differentiable through the bridge.",
                "polar": "SciPy bridge; not differentiable through the bridge.",
                "quadratic_assignment": "SciPy bridge; returns Python indices.",
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
    "EvaluationUtilities": {
        "numpy": "supported",
        "pytorch": "partial",
        "jax": "partial",
        "notes": "Some plotting, assignment, and summary operations remain NumPy/SciPy oriented.",
    },
}

BACKEND_SUPPORT_LEVELS: Final = ("supported", "partial", "unsupported")


def get_unsupported_functions(backend_name: str, module_name: str = "") -> tuple[str, ...]:
    """Return unsupported facade functions for a backend module."""
    backend = BACKEND_CAPABILITIES.get(backend_name, {})
    unsupported = backend.get("unsupported", {})
    return tuple(unsupported.get(module_name, ()))


def get_partial_capabilities(backend_name: str, module_name: str = "") -> dict[str, str]:
    """Return partial-support notes for a backend module."""
    backend = BACKEND_CAPABILITIES.get(backend_name, {})
    partial = backend.get("partial", {})
    return dict(partial.get(module_name, {}))



def get_api_backend_support(api_name: str) -> dict[str, str]:
    """Return backend support metadata for a public API name."""
    return dict(API_BACKEND_CAPABILITIES.get(api_name, {}))


def iter_api_backend_capabilities() -> tuple[tuple[str, dict[str, str]], ...]:
    """Return public API backend support rows in a stable order."""
    return tuple(sorted(API_BACKEND_CAPABILITIES.items()))
