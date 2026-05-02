# Public Protocols

`pyrecest.protocols` is the public home for small capability contracts used by
PyRecEst components.

A protocol describes what an object can do without requiring it to inherit from
a specific base class. This keeps extension points lightweight: a user-defined
class can work with generic PyRecEst utilities when it implements the required
methods and attributes.

## Current scope

The initial protocol package defines common dimension protocols, broad array
aliases, and model capability protocols:

- `SupportsDim` for objects with an intrinsic state-space dimension;
- `SupportsInputDim` for objects with an ambient or input coordinate dimension;
- `ArrayLike` and `BackendArray` as intentionally broad aliases for backend
  compatible values;
- `pyrecest.protocols.models` for likelihood, transition, prediction, and
  linear-Gaussian model capabilities.

Follow-up pull requests can add distribution, filter, conversion, and
manifold-specific protocols independently.

## Design principles

Protocols should stay small and capability-oriented. Instead of requiring every
distribution, model, or filter to implement one large interface, PyRecEst should
ask only for the capability that a function actually needs.

For example, a future density utility may require a `SupportsPdf` protocol while
a sampler utility may require only `SupportsSampling`. A particle representation
should not need to implement analytic density evaluation merely to satisfy a
large distribution base interface.

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
```

Package-level exports are intentionally minimal in the early protocol pull
requests to reduce merge conflicts while follow-up protocol modules are developed
in parallel.

## Model protocols

Model protocols live in `pyrecest.protocols.models`:

```python
from pyrecest.protocols.models import SupportsLikelihood, SupportsTransitionSampling
```

They cover small model capabilities such as:

- `SupportsLikelihood` for objects exposing `likelihood(measurement, state)`;
- `SupportsLogLikelihood` for log-domain measurement likelihoods;
- `SupportsTransitionSampling` for objects exposing `sample_next(state, n=1)`;
- `SupportsTransitionDensity` for transition-density evaluators;
- `SupportsPredictedDistribution` for models that can propagate a distribution;
- `SupportsLinearGaussianTransition` and `SupportsLinearGaussianMeasurement` for
  explicit linear-Gaussian model objects.

The expanded model protocol set is re-exported from `pyrecest.models`. Existing
likelihood and transition protocols also remain re-exported from
`pyrecest.models.likelihood` for backwards compatibility. The preferred public
protocol import path for new code is the protocol submodule.
