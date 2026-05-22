"""Machine-readable public API stability registry."""

from __future__ import annotations

from typing import Final

PUBLIC_API_CATEGORIES: Final = (
    "stable",
    "experimental",
    "deprecated",
    "backend-specific",
)

PUBLIC_API_REGISTRY: Final = {
    "BackendFacade": {
        "module": "pyrecest.backend",
        "category": "backend-specific",
        "backend_contract": "BackendFacade",
        "notes": "Facade names are importable across backends, with bridged or unsupported functions documented in the backend matrix.",
    },
    "DiscreteStateUtilities": {
        "module": "pyrecest.filters",
        "category": "backend-specific",
        "backend_contract": "DiscreteStateUtilities",
        "notes": "Finite-state HMM and IMM utilities operate on NumPy arrays and SciPy sparse matrices.",
    },
    "KalmanFilter": {
        "module": "pyrecest.filters",
        "category": "stable",
        "backend_contract": "KalmanFilter",
        "notes": "Linear Gaussian filtering is part of the portable baseline.",
    },
    "UnscentedKalmanFilter": {
        "module": "pyrecest.filters",
        "category": "backend-specific",
        "backend_contract": "UnscentedKalmanFilter",
        "notes": "Portable for backend-compatible model functions; advanced paths may bridge through NumPy/SciPy.",
    },
    "EuclideanParticleFilter": {
        "module": "pyrecest.filters",
        "category": "backend-specific",
        "backend_contract": "EuclideanParticleFilter",
        "notes": "Particle behavior depends on sampler and resampling support in the active backend.",
    },
    "DistributionConversion": {
        "module": "pyrecest.distributions.conversion",
        "category": "backend-specific",
        "backend_contract": "DistributionConversion",
        "notes": "Euclidean Gaussian/particle routes are portable; grid, Fourier, and manifold routes are route-specific.",
    },
    "UKFOnManifolds": {
        "module": "pyrecest.filters",
        "category": "backend-specific",
        "backend_contract": "UKFOnManifolds",
        "notes": "Current predict/update paths explicitly exclude JAX.",
    },
    "SphericalHarmonicsEOTTracker": {
        "module": "pyrecest.filters",
        "category": "backend-specific",
        "backend_contract": "SphericalHarmonicsEOTTracker",
        "notes": "Depends on spherical-harmonics and SciPy-adjacent functionality.",
    },
    "GaussianDistribution": {
        "module": "pyrecest.distributions",
        "category": "stable",
        "backend_contract": "GaussianDistribution",
        "notes": "Basic construction, moment access, and portable operations are part of the core distribution API.",
    },
    "LinearDiracDistribution": {
        "module": "pyrecest.distributions",
        "category": "stable",
        "backend_contract": "LinearDiracDistribution",
        "notes": "Core particle-style representation used by conversion and filtering workflows.",
    },
    "MultiBernoulliTracker": {
        "module": "pyrecest.filters",
        "category": "backend-specific",
        "backend_contract": "MultiBernoulliTracker",
        "notes": "Tracking workflows rely on assignment and measurement-set utilities with NumPy-oriented paths.",
    },
    "PointSetRegistration": {
        "module": "pyrecest.utils",
        "category": "backend-specific",
        "backend_contract": "PointSetRegistration",
        "notes": "Registration helpers may bridge through NumPy/SciPy and are not guaranteed differentiable.",
    },
    "EvaluationUtilities": {
        "module": "pyrecest.evaluation",
        "category": "backend-specific",
        "backend_contract": "EvaluationUtilities",
        "notes": "Plotting, assignment, summaries, and result helpers are only partly backend-portable.",
    },
}


def get_public_api_registry_entry(api_name: str) -> dict[str, str]:
    """Return a copy of one public API registry row."""
    return dict(PUBLIC_API_REGISTRY.get(api_name, {}))


def iter_public_api_registry() -> tuple[tuple[str, dict[str, str]], ...]:
    """Return public API registry rows in stable name order."""
    return tuple(sorted(PUBLIC_API_REGISTRY.items()))
