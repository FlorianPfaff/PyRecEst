# Public Protocols

`pyrecest.protocols` is the public home for small capability contracts used by
PyRecEst components.

A protocol describes what an object can do without requiring it to inherit from
a specific base class. This keeps extension points lightweight: a user-defined
class can work with generic PyRecEst utilities when it implements the required
methods and attributes.

## Current scope

The package currently defines common dimension protocols, broad array aliases,
and manifold/state-space protocols:

- `SupportsDim` for objects with an intrinsic state-space dimension;
- `SupportsInputDim` for objects with an ambient or input coordinate dimension;
- `ArrayLike` and `BackendArray` as intentionally broad aliases for backend
  compatible values;
- `StateSpaceLike`, `ManifoldLike`, and `EmbeddedManifoldLike` for objects that
  expose intrinsic and input dimensions;
- `SupportsManifoldSize` and `FiniteMeasureManifoldLike` for compact or bounded
  domains with a finite total measure;
- operation-level protocols such as `SupportsDistance`,
  `SupportsCoordinateNormalization`, `SupportsAngularError`,
  `SupportsDomainIntegration`, `SupportsDomainFunctionIntegration`,
  `SupportsIntegrationBoundaries`, and
  `SupportsHypersphericalCoordinateConversion`.

Follow-up pull requests can add distribution, filter, model, and conversion
protocols independently.

## Design principles

Protocols should stay small and capability-oriented. Instead of requiring every
distribution, model, or filter to implement one large interface, PyRecEst should
ask only for the capability that a function actually needs.

For example, a future density utility may require a `SupportsPdf` protocol while
a sampler utility may require only `SupportsSampling`. A particle representation
should not need to implement analytic density evaluation merely to satisfy a
large distribution base interface.

Manifold protocols follow the same principle. `ManifoldLike` requires only
`dim` and `input_dim`; finite domain size, distance evaluation, coordinate
normalization, angular error, and domain integration are separate capabilities.
A non-compact Euclidean state space can be manifold-like without supporting
`get_manifold_size()`.

## Runtime checks

The public protocols are runtime-checkable where practical:

```python
from pyrecest.protocols.common import SupportsDim


class DemoObject:
    dim = 2


assert isinstance(DemoObject(), SupportsDim)
```

Manifold-specific protocols are imported from their submodule:

```python
from pyrecest.protocols.manifolds import FiniteMeasureManifoldLike


class CircularDomain:
    dim = 1
    input_dim = 1

    def get_manifold_size(self):
        return 2.0 * 3.141592653589793

    def get_ln_manifold_size(self):
        return 1.8378770664093453


assert isinstance(CircularDomain(), FiniteMeasureManifoldLike)
```

Runtime checks confirm that the required attributes or methods are present. They
do not prove mathematical correctness. Protocol-specific tests should check
shapes, backend behavior, and semantics separately.

## Import style

Use submodule imports in early protocol pull requests:

```python
from pyrecest.protocols.common import SupportsDim, SupportsInputDim
from pyrecest.protocols.manifolds import ManifoldLike, SupportsManifoldSize
```

Package-level exports are intentionally minimal in this seed package to reduce
merge conflicts while follow-up protocol modules are developed in parallel.
