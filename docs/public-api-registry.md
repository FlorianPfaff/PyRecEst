# Public API Registry

The public API registry is the set of names exported by package-level namespaces
such as `pyrecest.filters`. It complements the backend API matrix: the registry
answers "what name is public?" and the matrix answers "which backend supports
that name?"

The machine-readable source is `src/pyrecest/api_registry.py`.

Run this check after adding, removing, stabilizing, deprecating, or reclassifying
user-facing APIs:

```bash
PYTHONPATH=src python scripts/check_public_api_registry.py --check docs/public-api-registry.md
```

## Rules

1. Prefer one canonical spelling for each public object.
2. Keep compatibility aliases only when they protect existing user code or a
   documented external naming convention.
3. Add every new backend-sensitive object to `docs/backend-api-matrix.md` and
   `src/pyrecest/_backend/capabilities.py`.
4. Keep lazy export maps, `__all__`, and documentation synchronized with tests.

## Filters Namespace

`pyrecest.filters` is lazy: importing the namespace does not import every
tracker implementation. Add new filter symbols to `_FILTER_EXPORTS` in
`src/pyrecest/filters/__init__.py`; `__all__` is generated from that map.

Compatibility aliases such as mixed acronym/camelcase tracker names should stay
mapped to the same implementation module as their canonical form. New examples
and documentation should use the canonical spelling.

<!-- public-api-registry:start -->
| API                            | Module                              | Category         | Backend contract               | Notes                                                                                                                |
|--------------------------------|-------------------------------------|------------------|--------------------------------|----------------------------------------------------------------------------------------------------------------------|
| `BackendFacade`                | `pyrecest.backend`                  | backend-specific | `BackendFacade`                | Facade names are importable across backends, with bridged or unsupported functions documented in the backend matrix. |
| `DiscreteStateUtilities`       | `pyrecest.filters`                  | backend-specific | `DiscreteStateUtilities`       | Finite-state HMM and IMM utilities operate on NumPy arrays and SciPy sparse matrices.                                |
| `DistributionConversion`       | `pyrecest.distributions.conversion` | backend-specific | `DistributionConversion`       | Euclidean Gaussian/particle routes are portable; grid, Fourier, and manifold routes are route-specific.              |
| `EuclideanParticleFilter`      | `pyrecest.filters`                  | backend-specific | `EuclideanParticleFilter`      | Particle behavior depends on sampler and resampling support in the active backend.                                   |
| `EvaluationUtilities`          | `pyrecest.evaluation`               | backend-specific | `EvaluationUtilities`          | Plotting, assignment, summaries, and result helpers are only partly backend-portable.                                |
| `GaussianDistribution`         | `pyrecest.distributions`            | stable           | `GaussianDistribution`         | Basic construction, moment access, and portable operations are part of the core distribution API.                    |
| `KalmanFilter`                 | `pyrecest.filters`                  | stable           | `KalmanFilter`                 | Linear Gaussian filtering is part of the portable baseline.                                                          |
| `LinearDiracDistribution`      | `pyrecest.distributions`            | stable           | `LinearDiracDistribution`      | Core particle-style representation used by conversion and filtering workflows.                                       |
| `MultiBernoulliTracker`        | `pyrecest.filters`                  | backend-specific | `MultiBernoulliTracker`        | Tracking workflows rely on assignment and measurement-set utilities with NumPy-oriented paths.                       |
| `PointSetRegistration`         | `pyrecest.utils`                    | backend-specific | `PointSetRegistration`         | Registration helpers may bridge through NumPy/SciPy and are not guaranteed differentiable.                           |
| `SphericalHarmonicsEOTTracker` | `pyrecest.filters`                  | backend-specific | `SphericalHarmonicsEOTTracker` | Depends on spherical-harmonics and SciPy-adjacent functionality.                                                     |
| `UKFOnManifolds`               | `pyrecest.filters`                  | backend-specific | `UKFOnManifolds`               | Current predict/update paths explicitly exclude JAX.                                                                 |
| `UnscentedKalmanFilter`        | `pyrecest.filters`                  | backend-specific | `UnscentedKalmanFilter`        | Portable for backend-compatible model functions; advanced paths may bridge through NumPy/SciPy.                      |
<!-- public-api-registry:end -->
