# Model Objects

Model objects describe transition and measurement capabilities independently of a concrete filter implementation.

This initial model layer focuses on likelihood- and sampling-based capabilities for particle filters, grid filters, and other callback-based estimators.

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

## Protocol-based adapters

`pyrecest.models` also exposes small adapter helpers that consume public model protocols from `pyrecest.protocols.models`.

Use the `as_*` helpers when a caller may pass either an existing model object or a plain callback:

```python
from pyrecest.models import as_likelihood_model, evaluate_likelihood

measurement_model = as_likelihood_model(likelihood)
weight = evaluate_likelihood(measurement_model, measurement, state)
```

The same pattern is available for sampleable and density-based transition models:

```python
from pyrecest.models import (
    as_density_transition_model,
    as_sampleable_transition_model,
    evaluate_transition_density,
    sample_next_state,
)

sampler_model = as_sampleable_transition_model(sample_next)
samples = sample_next_state(sampler_model, state, n=100)

density_model = as_density_transition_model(transition_density)
weights = evaluate_transition_density(density_model, state_next, state_previous)
```

Linear Gaussian model objects can be adapted to the argument names expected by existing linear Kalman-style APIs:

```python
from pyrecest.models import linear_measurement_arguments, linear_transition_arguments

prediction_args = linear_transition_arguments(transition_model)
update_args = linear_measurement_arguments(measurement_model)

kalman_filter.predict_linear(
    prediction_args.system_matrix,
    prediction_args.sys_noise_cov,
    prediction_args.sys_input,
)
kalman_filter.update_linear(
    measurement,
    update_args.measurement_matrix,
    update_args.meas_noise,
)
```

These helpers are intentionally structural. They accept canonical names and existing aliases such as `system_noise_cov` / `sys_noise_cov` and `measurement_noise_cov` / `meas_noise`.

## Scope

These model objects and adapters are additive infrastructure. They do not deprecate existing filter APIs and do not modify filter behavior by themselves. Filter-specific adapters can consume these objects in later changes.
