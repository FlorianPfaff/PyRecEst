# Protocol Capability Matrix

This page records a smoke-test-backed snapshot of selected public PyRecEst
classes and the protocol-style capabilities they currently provide.

The matrix is intentionally conservative:

- it checks a small representative subset of public classes;
- it records current behavior, not planned behavior;
- it treats methods that are present but raise `NotImplementedError` or an
  unsupported-operation `ValueError` as unsupported for that capability;
- it does not prove mathematical correctness, numerical accuracy, or backend
  parity.

The corresponding tests live in `tests/protocols/` and should be updated when a
class intentionally gains or loses one of these capabilities.

## Distribution capabilities

| Class                         | `dim` | `input_dim` | `pdf` | `ln_pdf` | `sample` | `mean` | `covariance` | `convert_to` | `approximate_as` | `from_distribution` |
|-------------------------------|------:|------------:|------:|---------:|---------:|-------:|-------------:|-------------:|-----------------:|--------------------:|
| `GaussianDistribution`        |   yes |         yes |   yes |      yes |      yes |    yes |          yes |          yes |              yes |                 yes |
| `LinearDiracDistribution`     |   yes |         yes |    no |       no |      yes |    yes |          yes |          yes |              yes |                 yes |
| `VonMisesDistribution`        |   yes |         yes |   yes |      yes |      yes |    yes |           no |          yes |              yes |                  no |
| `CircularUniformDistribution` |   yes |         yes |   yes |      yes |      yes |     no |           no |          yes |              yes |                  no |

Notes:

- `LinearDiracDistribution` has a `pdf` method through the current class
  hierarchy, but the method is intentionally undefined for a continuous density
  interpretation and raises `NotImplementedError`.
- `CircularUniformDistribution.mean()` is not considered supported because the
  circular uniform distribution has no unique mean direction.
- `convert_to` and `approximate_as` are checked as structural conversion-entry
  capabilities. The matrix does not claim that every target representation is
  available for every source distribution.

## Filter capabilities

| Class                    | `dim` | `filter_state` | `get_point_estimate` | `predict_linear` | `update_linear` | `predict_nonlinear` | `update_nonlinear` | `predict_model` | `update_model` | `update_nonlinear_using_likelihood` | history recording |
|--------------------------|------:|---------------:|---------------------:|-----------------:|----------------:|--------------------:|-------------------:|----------------:|---------------:|------------------------------------:|------------------:|
| `KalmanFilter`           |   yes |            yes |                  yes |              yes |             yes |                  no |                 no |             yes |            yes |                                  no |               yes |
| `UnscentedKalmanFilter`  |   yes |            yes |                  yes |              yes |             yes |                 yes |                yes |             yes |            yes |                                  no |               yes |
| `CircularParticleFilter` |   yes |            yes |                  yes |               no |              no |                 yes |                 no |             yes |            yes |                                 yes |               yes |

Notes:

- Filter capabilities are checked at class level to avoid running prediction or
  update algorithms in this smoke-test layer.
- `history recording` means that both `record_filter_state` and
  `record_point_estimate` are available.
- Backend-specific algorithm support is documented separately in the backend
  compatibility guide; this matrix only records the public entry points exposed
  by the classes.

## Maintenance rule

When a new public protocol module lands, keep this matrix aligned with it. For
example, when distribution-specific protocols such as `SupportsPdf` or
`SupportsSampling` are introduced, this page and the tests should use the same
capability names and semantics.
