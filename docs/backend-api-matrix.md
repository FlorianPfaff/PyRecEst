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
```

## Support Levels

| Level         | Meaning                                                                                                                       |
|---------------|-------------------------------------------------------------------------------------------------------------------------------|
| `supported`   | Intended to preserve backend semantics for the listed API.                                                                    |
| `partial`     | Numerically useful, but with documented limitations such as SciPy bridges, CPU copies, or missing gradient/device guarantees. |
| `unsupported` | Should raise a clear `NotImplementedError` or be documented as unavailable for the backend.                                   |

## Public API Rows

| API                            | NumPy     | PyTorch     | JAX         | Notes                                                                                                        |
|--------------------------------|-----------|-------------|-------------|--------------------------------------------------------------------------------------------------------------|
| `KalmanFilter`                 | supported | supported   | supported   | Linear Gaussian operations are the portable baseline.                                                        |
| `UKFOnManifolds`               | supported | partial     | unsupported | JAX exclusions are currently explicit.                                                                       |
| `SphericalHarmonicsEOTTracker` | supported | unsupported | unsupported | Depends on spherical harmonics and SciPy-adjacent functionality.                                             |
| `GaussianDistribution`         | supported | supported   | supported   | Basic construction and portable operations should stay backend portable.                                     |
| `LinearDiracDistribution`      | supported | supported   | supported   | Used by conversion and particle-style workflows.                                                             |
| `UnscentedKalmanFilter`        | supported | partial     | partial     | Portable for backend-compatible model functions; advanced paths may still bridge through NumPy/SciPy.        |
| `EuclideanParticleFilter`      | supported | partial     | partial     | Particle operations are portable where sampling and resampling helpers preserve backend semantics.           |
| `DistributionConversion`       | supported | partial     | partial     | Euclidean particle/Gaussian conversions are portable; grid, Fourier, and manifold routes are route-specific. |
| `MultiBernoulliTracker`        | supported | partial     | unsupported | Tracking workflows rely on assignment and measurement-set utilities that are currently NumPy-oriented.       |
| `PointSetRegistration`         | supported | partial     | unsupported | Registration utilities may copy through NumPy/SciPy and should not be assumed differentiable.                |
| `EvaluationUtilities`          | supported | partial     | partial     | Plotting, assignment, and summaries remain partly NumPy/SciPy oriented.                                      |

When adding a new public API, add a row to the matrix, update docs if the row is
user-facing, and add a focused backend test if the API is expected to be
portable.
