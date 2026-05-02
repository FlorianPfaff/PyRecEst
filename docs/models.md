# Model Objects

`pyrecest.models` provides reusable, filter-independent contracts for transition,
measurement, and noise models.

A filter estimates a state. A model describes how states evolve and how
measurements are generated. Keeping those concepts separate makes it possible to
reuse the same mathematical model with multiple compatible filters and with
simulation or evaluation code.

## Current scope

The model package currently defines marker base classes, metadata, and small
capability protocols. It is intentionally additive: existing filter APIs such as
`predict_linear(...)`, `update_linear(...)`, `predict_nonlinear(...)`, and
`update_nonlinear(...)` remain unchanged.

Future adapters can consume these model capabilities through methods such as
`predict_model(...)` and `update_model(...)` without changing the established
low-level filter methods.

## Marker classes

The base marker classes are:

- `Model`
- `TransitionModel`
- `MeasurementModel`
- `NoiseModel`

They carry optional `ModelMetadata` such as `name`, `state_dim`, and
`measurement_dim`.

## Capability protocols

Model behavior is described through small runtime-checkable protocols. A model
may implement one or more of them:

- `HasLinearTransition` for linear predictions with `system_matrix`,
  `system_noise_cov`, and optional `sys_input`.
- `HasLinearMeasurement` for linear updates with `measurement_matrix` and
  `meas_noise`.
- `HasTransitionFunction` and `HasMeasurementFunction` for deterministic
  nonlinear mappings.
- `HasTransitionJacobian` and `HasMeasurementJacobian` for linearization-based
  filters.
- `HasTransitionSampler` for particle filters and simulation.
- `HasTransitionDensity` for density-based prediction.
- `HasLikelihood` for likelihood-based measurement updates.

## Example

```python
from pyrecest.models import HasLinearTransition, ModelMetadata, TransitionModel


class ConstantVelocityTransition(TransitionModel):
    metadata = ModelMetadata(name="constant velocity", state_dim=2)
    system_matrix = ((1.0, 1.0), (0.0, 1.0))
    system_noise_cov = ((0.1, 0.0), (0.0, 0.1))
    sys_input = None


transition_model = ConstantVelocityTransition()
assert isinstance(transition_model, HasLinearTransition)
```

This example only defines the model object. Filter adapters that consume these
model capabilities can be added independently.
