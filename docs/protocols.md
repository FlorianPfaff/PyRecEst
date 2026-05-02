# Public Protocols

`pyrecest.protocols` is the public home for small capability contracts used by
PyRecEst components.

A protocol describes what an object can do without requiring it to inherit from
a specific base class. This keeps extension points lightweight: a user-defined
class can work with generic PyRecEst utilities when it implements the required
methods and attributes.

## Current scope

The protocol package currently defines common dimension protocols, broad array
aliases, and model capability protocols:

- `SupportsDim` for objects with an intrinsic state-space dimension;
- `SupportsInputDim` for objects with an ambient or input coordinate dimension;
- `ArrayLike` and `BackendArray` as intentionally broad aliases for backend
  compatible values;
- `SupportsLikelihood` and `SupportsLogLikelihood` for measurement models;
- `SupportsTransitionSampling` and `SupportsTransitionDensity` for transition
  models;
- `SupportsPredictedDistribution` for models that propagate distribution
  objects directly;
- `SupportsLinearGaussianTransition` and `SupportsLinearGaussianMeasurement` for
  canonical linear Gaussian model objects.

Follow-up pull requests can add distribution, filter, conversion, and
manifold-specific protocols independently.

## Design principles

Protocols should stay small and capability-oriented. Instead of requiring every
distribution, model, or filter to implement one large interface, PyRecEst should
ask only for the capability that a function actually needs.

For example, a density utility may require a `SupportsPdf` protocol while a
sampler utility may require only `SupportsSampling`. A particle representation
should not need to implement analytic density evaluation merely to satisfy a
large distribution base interface.

## Model protocols

Model protocols are available from `pyrecest.protocols.models`:

```python
from pyrecest.protocols.models import SupportsLikelihood


def use_likelihood(model: SupportsLikelihood, measurement, state):
    return model.likelihood(measurement, state)
```

Objects do not have to inherit from a PyRecEst base class. Any object with the
required methods satisfies the protocol structurally:

```python
class DemoMeasurementModel:
    def likelihood(self, measurement, state):
        return measurement + state


assert isinstance(DemoMeasurementModel(), SupportsLikelihood)
```

## Runtime checks

The public protocols are runtime-checkable where practical:

```python
from pyrecest.protocols.common import SupportsDim


class DemoObject:
    dim = 2


assert isinstance(DemoObject(), SupportsDim)
```

Runtime checks confirm that the required attributes or methods are present. They
do not prove mathematical correctness. Protocol-specific tests should check
shapes, backend behavior, and semantics separately.

## Import style

Use submodule imports in early protocol pull requests:

```python
from pyrecest.protocols.common import SupportsDim, SupportsInputDim
from pyrecest.protocols.models import SupportsLikelihood
```

Package-level exports are intentionally minimal in the seed package to reduce
merge conflicts while follow-up protocol modules are developed in parallel.
