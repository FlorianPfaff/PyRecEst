# Backend API Matrix

PyRecEst has two related backend contracts:

1. the facade-level contract for functions exposed through `pyrecest.backend`;
2. the public API contract for distributions, filters, trackers, and utilities.

The machine-readable source for both contracts is
`src/pyrecest/_backend/capabilities.py`.

To inspect the current matrix from a checkout or installed environment, run:

```bash
pyrecest backends --format markdown
python scripts/render_backend_api_matrix.py
python scripts/check_backend_api_matrix.py
```

The documentation table is checked against `src/pyrecest/_backend/capabilities.py`
in CI so the user-facing matrix cannot silently drift from the executable metadata.

## Support Levels

| Level         | Meaning                                                                                                                       |
|---------------|-------------------------------------------------------------------------------------------------------------------------------|
| `supported`   | Intended to preserve backend semantics for the listed API.                                                                    |
| `bridged`     | Works by crossing into another numerical stack, usually NumPy/SciPy; do not assume device, dtype, or gradient preservation.   |
| `partial`     | Numerically useful, but with documented limitations such as SciPy bridges, CPU copies, or missing gradient/device guarantees. |
| `unsupported` | Should raise a clear `NotImplementedError` or be documented as unavailable for the backend.                                   |

## Public API Rows

<!-- backend-api-matrix:start -->
| API                            | NumPy     | PyTorch     | JAX         | Notes                                                                                                                            |
|--------------------------------|-----------|-------------|-------------|----------------------------------------------------------------------------------------------------------------------------------|
| `BackendFacade`                | supported | partial     | partial     | Facade names are importable across backends, but some functions are bridged or explicitly unsupported.                           |
| `DiscreteStateUtilities`       | supported | bridged     | bridged     | Finite-state HMM and IMM utilities operate on NumPy arrays and SciPy sparse matrices; non-NumPy inputs are coerced.              |
| `DistributionConversion`       | supported | partial     | partial     | Euclidean particle/Gaussian conversions are portable; grid, Fourier, and manifold routes are route-specific.                     |
| `EuclideanParticleFilter`      | supported | partial     | partial     | Particle operations are portable where sampling and resampling helpers preserve backend semantics.                               |
| `EvaluationUtilities`          | supported | bridged     | bridged     | Some plotting, assignment, and summary operations remain NumPy/SciPy oriented and may not preserve device or gradient semantics. |
| `GaussianDistribution`         | supported | supported   | supported   | Basic construction, moment access, and portable operations should remain backend portable.                                       |
| `KalmanFilter`                 | supported | supported   | supported   | Linear Gaussian operations are part of the portable baseline.                                                                    |
| `LinearDiracDistribution`      | supported | supported   | supported   | Used by representation conversion and particle-style workflows.                                                                  |
| `MultiBernoulliTracker`        | supported | partial     | unsupported | Tracking workflows rely on assignment and measurement-set utilities that are currently NumPy-oriented.                           |
| `PointSetRegistration`         | supported | partial     | unsupported | Registration utilities may copy through NumPy/SciPy and should not be assumed differentiable.                                    |
| `SphericalHarmonicsEOTTracker` | supported | unsupported | unsupported | Depends on spherical harmonics and SciPy-adjacent functionality.                                                                 |
| `UKFOnManifolds`               | supported | partial     | unsupported | The current implementation documents explicit JAX exclusions for predict/update.                                                 |
| `UnscentedKalmanFilter`        | supported | partial     | partial     | Portable for backend-compatible model functions; advanced paths may still bridge through NumPy/SciPy.                            |
<!-- backend-api-matrix:end -->

When adding a new public API, add a row to the matrix, update docs if the row is
user-facing, add or update the generated table, and add a focused backend test
if the API is expected to be portable. CI checks that this table still reflects
`src/pyrecest/_backend/capabilities.py`.

## Runtime Access

Use the public helper when examples or downstream packages need to inspect
backend support without duplicating the table:

```python
from pyrecest import get_backend_support

assert get_backend_support("KalmanFilter", backend="jax") == "supported"
```

The CLI can also render the matrix:

```bash
pyrecest backends --format markdown
```
