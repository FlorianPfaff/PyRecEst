# Model Objects

Model objects describe transition and measurement capabilities independently of a concrete filter implementation.

This initial model layer focuses on likelihood- and sampling-based capabilities for particle filters, grid filters, and other callback-based estimators. Public model capability protocols are available from `pyrecest.protocols.models`.

## Measurement likelihoods

Use `LikelihoodMeasurementModel` when a measurement update is represented by a likelihood callback:

```python
from pyrecest.models import LikelihoodMeasurementModel

measurement_model = LikelihoodMeasurementModel(likelihood)
weight = measurement_model.likelihood(measurement, state)
```

The callback convention is `likelihood(measurement, state)`.

A `log_likelihood` callback may also be supplied when the log-domain form is preferable.

## State-conditioned distributions

`LikelihoodMeasurementModel.from_distribution_factory(...)` builds a likelihood model from a callable that returns a conditional measurement distribution for a state. The returned distribution must expose a density method such as `pdf`.

## Sampleable transitions

Use `SampleableTransitionModel` when the transition model can draw next-state samples:

```python
from pyrecest.models import SampleableTransitionModel

transition_model = SampleableTransitionModel(sample_next)
samples = transition_model.sample_next(state, n=100)
```

The callback convention is `sample_next(state, n=1)`.

## Density-based transitions

Use `DensityTransitionModel` when the transition model can evaluate a transition density:

```python
from pyrecest.models import DensityTransitionModel

transition_model = DensityTransitionModel(transition_density)
density = transition_model.transition_density(state_next, state_previous)
```

The callback convention is `transition_density(state_next, state_previous)`.

## Public model protocols

Use `pyrecest.protocols.models` when writing generic code that accepts any model
with a required capability:

```python
from pyrecest.protocols.models import SupportsLikelihood, SupportsTransitionSampling


def evaluate_weights(model: SupportsLikelihood, measurement, states):
    return model.likelihood(measurement, states)


def draw_predictions(model: SupportsTransitionSampling, state, n):
    return model.sample_next(state, n=n)
```

The same protocols remain importable from `pyrecest.models` for compatibility
with earlier model-layer code, but the protocol submodule is the preferred
public home for new code.

## Scope

These model objects are additive infrastructure. They do not deprecate existing filter APIs and do not modify filter behavior by themselves. Filter-specific adapters can consume these objects in later changes.
