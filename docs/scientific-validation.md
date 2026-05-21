# Scientific Validation

PyRecEst tests should cover more than importability and API shape. Recursive
Bayesian estimation code also needs validation against mathematical invariants,
known statistical diagnostics, and backend-specific numerical behavior.

This page defines the validation ladder used when adding or changing filters,
distributions, samplers, trackers, or evaluation helpers.

Scientific regressions are often subtler than import or shape failures. Prefer
invariant checks over brittle full-trajectory comparisons unless the scenario is
intentionally deterministic and has a documented golden output.

## Validation Layers

| Layer                          | Purpose                                                         | Examples                                                                                                        |
|--------------------------------|-----------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| API smoke tests                | Confirm public entry points exist and have stable capabilities. | Protocol capability matrices, import checks, CLI smoke tests.                                                   |
| Deterministic algebraic checks | Verify identities that should hold without randomness.          | Gaussian multiplication, Kalman covariance symmetry, normalized innovation squared consistency.                 |
| Numerical invariant checks     | Catch invalid estimates even when exact values are not known.   | Positive semidefinite covariances, nonnegative probabilities, normalized weights, unit-norm directional states. |
| Monte Carlo checks             | Verify statistical behavior across repeated randomized runs.    | NEES/NIS coverage, sampling moment convergence, resampling effective sample size behavior.                      |
| Scenario regression checks     | Preserve known behavior for complete workflows.                 | Scenario zoo expected outputs, benchmark regressions, tracker association edge cases.                           |

## Core Invariants

| Component                           | Invariant                                                                      |
|-------------------------------------|--------------------------------------------------------------------------------|
| Probability distributions           | Densities integrate or sum to one on their manifold.                           |
| Circular and toroidal distributions | Wrapping by the period preserves density.                                      |
| Hyperspherical distributions        | Samples and support points remain unit norm.                                   |
| Gaussian filters                    | Covariances stay symmetric positive definite after updates.                    |
| Particle filters                    | Weights remain finite, non-negative, and normalized after resampling.          |
| Representation conversion           | Moment-matching routes preserve mean/covariance within tolerance.              |
| Trackers                            | Cardinality, association, and gating diagnostics remain internally consistent. |

When a change affects a Kalman-style Gaussian estimator, check at least:

- covariance matrices remain symmetric after prediction and update;
- covariance matrices remain positive semidefinite up to numerical tolerance;
- normalized innovation squared values are nonnegative and agree with the
  innovation covariance solve;
- rejected gated measurements leave the posterior unchanged;
- diagnostics report the covariance scale and update action used by robust
  updates.

When a change affects particle or grid methods, check at least:

- weights remain finite and normalized;
- resampling never creates invalid particle shapes;
- likelihood-only updates handle zero or underflowing likelihoods explicitly;
- deterministic seeds are recorded for reproducibility.

When a change affects circular, spherical, or manifold-valued states, check at
least:

- wrapped coordinates remain in the documented convention;
- unit-vector or quaternion states remain normalized;
- antipodal or periodic equivalences are tested where the distribution assumes
  them;
- moment and point estimates are invariant under representation-specific
  symmetries.

## Test Placement

Use the most specific existing test directory when possible:

- `tests/filters/` for filter and tracker invariants;
- `tests/distributions/` for density, sampling, and conversion invariants;
- `tests/protocols/` for API capability snapshots;
- `tests/scenarios/` or scenario fixtures for complete reproducible workflows.

Mark slower randomized coverage with `@pytest.mark.numerical_stress` so the fast
matrix can remain focused while scheduled or manual runs exercise the heavier
statistical checks.

## Backend Expectations

For APIs listed as `supported` in the backend API matrix, add or update focused
tests that run under the NumPy, PyTorch, and JAX CI matrix. For APIs listed as
`partial`, test the portable subset and document what is intentionally excluded.
For `unsupported` APIs, prefer a clear unsupported-backend exception or
`NotImplementedError` path.

Backend-specific tolerances are acceptable, but they should be explicit in the
test and justified by dtype, device, tracing, or bridge behavior rather than by
an unexplained broad tolerance.
