# Model Objects

Model objects describe the state-transition and measurement parts of an
estimation problem independently from the filter that consumes them.

The additive-noise nonlinear models represent the two common forms

```text
x_next = f(x) + w
z      = h(x) + v
```

where `f` and `h` are deterministic functions and `w` and `v` are additive noise
terms. The noise object can be any PyRecEst-compatible distribution or user
object that exposes the capabilities needed by a particular method.

## Transition model

```python
from pyrecest.backend import array, diag
from pyrecest.distributions import GaussianDistribution
from pyrecest.models import AdditiveNoiseTransitionModel

transition_model = AdditiveNoiseTransitionModel(
    lambda x: array([x[0] + 1.0, 2.0 * x[1]]),
    noise_distribution=GaussianDistribution(array([0.0, 0.0]), diag(array([0.1, 0.2]))),
    jacobian=lambda x: diag(array([1.0, 2.0])),
)
```

Useful capabilities include:

- `transition_function(state)` for the noise-free propagation;
- `mean(state)` for propagation plus additive noise mean when available;
- `noise_covariance` for Gaussian-style filters;
- `jacobian(state)` for linearized filters;
- `sample_next(state, n)` for particle-style filters;
- `transition_density(next_state, state)` for density-based filters.

## Measurement model

```python
from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution
from pyrecest.models import AdditiveNoiseMeasurementModel

measurement_model = AdditiveNoiseMeasurementModel(
    lambda x: array([x[0] * x[0]]),
    noise_distribution=GaussianDistribution(array([0.0]), array([[1.0]])),
    jacobian=lambda x: array([[2.0 * x[0]]]),
)
```

Useful capabilities include:

- `measurement_function(state)` for the noise-free prediction;
- `predict_measurement(state)` for prediction plus additive noise mean when available;
- `noise_covariance` for Gaussian-style filters;
- `jacobian(state)` for linearized filters;
- `sample_measurement(state, n)` for simulation and particle-style methods;
- `likelihood(measurement, state)` for likelihood-based updates.

The existing filter APIs remain unchanged. Adapter methods such as
`predict_model(...)` and `update_model(...)` can be added in later PRs without
changing these model classes.
