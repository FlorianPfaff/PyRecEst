# API Overview

PyRecEst exposes most user-facing classes through package-level imports such as
`pyrecest.distributions`, `pyrecest.filters`, `pyrecest.models`,
`pyrecest.protocols`, `pyrecest.sampling`, `pyrecest.smoothers`,
`pyrecest.evaluation`, and `pyrecest.utils`.

Use backend-compatible arrays from `pyrecest.backend` in examples and user code
when the same code should run on more than one numerical backend.

```python
from pyrecest.backend import array, diag
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import KalmanFilter
from pyrecest.models import LikelihoodMeasurementModel
```

See [Shapes and conventions](conventions.md) for the expected vector, matrix,
batch, measurement-set, and manifold-coordinate shapes used by common APIs.

## Package Map

### `pyrecest.backend`

Dynamic backend facade used by the rest of the project. The default backend is
NumPy. PyTorch and JAX support are selected through `PYRECEST_BACKEND` before
importing `pyrecest`. See [backend compatibility](backend-compatibility.md) for
backend-specific support notes and known limitations.

### `pyrecest.protocols`

Public capability contracts for PyRecEst extension points. Protocols describe
what an object can do without requiring it to inherit from a specific PyRecEst
base class.

The current seed exposes common dimension protocols:

- `SupportsDim` for objects with an intrinsic state-space dimension;
- `SupportsInputDim` for objects with an ambient or input coordinate dimension.

See [public protocols](protocols.md), [custom distribution extensions](custom-distribution.md),
and [custom filter extensions](custom-filter.md) for protocol-oriented extension
examples.

### `pyrecest.distributions`

Probability distributions and density representations on linear spaces,
periodic spaces, spheres, hyperspheres, hypertori, Cartesian-product spaces,
and rigid-body state spaces.

Common starting points include:

- `GaussianDistribution` for Euclidean Gaussian states;
- `GaussianMixture` and `LinearMixture` for mixtures on linear spaces;
- `VonMisesDistribution` and wrapped distributions for circular states;
- `VonMisesFisherDistribution`, `BinghamDistribution`, and related classes for
  hyperspherical states;
- `HypertoroidalFourierDistribution` and hypertoroidal grid or wrapped-normal
  distributions for toroidal states.

Use `pyrecest.distributions.conversion.convert_distribution(...)` to convert
between analytic, Dirac/particle, grid, Fourier, and moment-matched
representations. The target may be a concrete class or an alias such as
`"particles"`, `"gaussian"`, `"grid"`, or `"fourier"`. See
[representation conversion](representation-conversion.md).

### `pyrecest.models`

Reusable transition and measurement model objects that describe model
capabilities independently of a concrete filter. The initial model layer focuses
on likelihood-based measurement models and transition models that can either
sample next states or evaluate transition densities.

Common starting points include:

- `LikelihoodMeasurementModel` for particle-filter or grid-filter style
  measurement likelihoods;
- `SampleableTransitionModel` for transition models that can draw samples from
  a next-state distribution;
- `DensityTransitionModel` for transition models that can evaluate transition
  densities.

See [model objects](models.md) for usage patterns and conventions.

### `pyrecest.filters`

Recursive Bayesian estimators and trackers. The package includes standard
linear and unscented Kalman filters, particle filters, grid filters,
manifold-specific filters, multi-target trackers, extended-object trackers, and
track-management utilities.

Common starting points include:

- `KalmanFilter` for linear Gaussian Euclidean state estimation;
- `AssociationHypothesis`, `NISGate`, `CostThresholdGate`, `TopKGate`,
  and `linear_gaussian_association_hypotheses` for reusable association
  diagnostics, gating, and cost-matrix conversion;
- `OutOfSequenceKalmanUpdater`, `OutOfSequenceParticleUpdater`,
  `FixedLagBuffer`, and `MeasurementTimeBuffer` for fixed-lag delayed
  measurement handling;
- `UnscentedKalmanFilter` and `UKFOnManifolds` for nonlinear models;
- `ManifoldExponentialMovingAverage` for streaming manifold-valued samples;
- `EuclideanParticleFilter` and manifold-specific particle filters;
- `VonMisesFilter`, `VonMisesFisherFilter`, and Fourier or grid filters for
  directional estimation;
- `MultiBernoulliTracker`, `GlobalNearestNeighbor`, `JPDAF`, and
  `TrackManager` for tracking workflows.

### `pyrecest.smoothers`

Backward-pass smoothers that refine filter estimates after a measurement
sequence has been processed.

Common starting points include:

- `RauchTungStriebelSmoother`;
- `UnscentedRauchTungStriebelSmoother`.

### `pyrecest.sampling`

Samplers and grid generators for Euclidean and manifold domains.

Common starting points include:

- `GaussianSampler`;
- `MerweScaledSigmaPoints` and `JulierSigmaPoints`;
- `FibonacciGridSampler`;
- `FibonacciRejectionSampler`;
- `SobolGridSampler`;
- `HaltonGridSampler`;
- `SphericalFibonacciSampler`;
- `HealpixHopfSampler`;
- `LeopardiSampler`;
- `get_grid_hypersphere`.

### `pyrecest.evaluation`

Simulation, measurement generation, evaluation, plotting, and result summary
helpers for filter and tracker experiments.

Common starting points include:

- `generate_groundtruth`;
- `generate_measurements`;
- `perform_predict_update_cycles`;
- `evaluate_for_variables`;
- `summarize_filter_results`;
- `plot_results`.

### `pyrecest.utils`

Reusable utility helpers for assignment, association models, history recording,
multi-session assignment, generic track-matrix evaluation, Gaussian/covariance
pairwise features, and point-set registration.

Common starting points include:

- `murty_k_best_assignments`;
- `LogisticPairwiseAssociationModel`;
- `pairwise_mahalanobis_distances`;
- `pairwise_covariance_shape_components`;
- `HistoryRecorder`;
- `solve_multisession_assignment`;
- `normalize_track_matrix`;
- `score_track_links`;
- `track_error_ledger`;
- `estimate_thin_plate_spline`.

## Where To Look Next

- Use [the examples guide](examples.md) for small runnable workflows.
- Use [public protocols](protocols.md) and the custom extension guides when you
  want user-defined objects to interoperate with generic PyRecEst utilities.
- Use [the generated API reference](reference/index.md) for package-level
  reference pages built from docstrings.
- Use `tests/` as executable reference coverage for APIs that do not yet have
  dedicated tutorials.
- Use module docstrings and class docstrings for detailed mathematical notes
  where they are available.
