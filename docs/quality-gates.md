# Quality Gates

PyRecEst is a research library, but public releases should still protect users
from accidental backend, packaging, documentation, and security regressions.
The repository therefore separates broad exploratory checks from the checks that
should be safe to require on every pull request.

## Required Pull Request Checks

Recommended required checks for protected branches are:

| Check                     | Purpose                                                                 |
|---------------------------|-------------------------------------------------------------------------|
| `Static analysis`         | Runs the static baseline, compile checks, and generated-doc checks.     |
| `Test workflow / docs`    | Builds documentation with `mkdocs build --strict`.                      |
| `Test workflow / package` | Builds distributions, installs the wheel, and runs smoke examples.      |
| `Test workflow / test`    | Runs the backend matrix for NumPy, PyTorch, and JAX.                    |
| `CodeQL`                  | Scans the Python codebase for security issues.                          |
| `Dependency review`       | Fails pull requests that introduce high-severity dependency advisories. |

Scheduled jobs may run larger or slower matrices, but the required checks should
remain small enough that contributors can iterate quickly.

## Backend Contract Changes

Any pull request that changes backend behavior should update the same source of
truth used by tests, documentation, and command-line inspection:

```text
src/pyrecest/_backend/capabilities.py
```

The generated backend API table in `docs/backend-api-matrix.md` must continue to
match that source. Public API category changes should also update
`src/pyrecest/api_registry.py` and `docs/public-api-registry.md`. Run these
commands locally after changing capability or public API registry rows:

```bash
PYTHONPATH=src python scripts/render_backend_api_matrix.py --check docs/backend-api-matrix.md
PYTHONPATH=src python scripts/check_public_api_registry.py --check docs/public-api-registry.md
```

If a public API is intended to be backend-portable, add a focused test that runs
on the relevant backend matrix. If it is intentionally backend-specific, add a
clear capability row and a user-facing failure mode.

## Static Analysis Baseline

The static workflow intentionally starts with an allowlist because parts of the
repository still contain historical re-export and backend-facade lint noise.
Expand the allowlist only when a module is clean under Ruff and mypy. Avoid
making the entire source tree required until the existing baseline is reconciled.

## Coverage And Benchmark Baselines

Coverage has a low initial floor so the project can enforce a real threshold
without blocking routine maintenance. Raise the threshold after adding tests to
backend contracts, representation conversion, filters, and tracker utilities.

Benchmarks should be deterministic and conservative. CI should fail on severe
slowdowns or numerical drift, not on small timing noise from shared runners.

## Dependency Footprint Changes

When moving dependencies into optional extras, keep the default installation
usable for the minimal Euclidean baseline:

- import `pyrecest`;
- construct a Gaussian distribution;
- run a linear Kalman predict/update cycle;
- build and install the wheel;
- keep error messages explicit for APIs that require optional extras.
