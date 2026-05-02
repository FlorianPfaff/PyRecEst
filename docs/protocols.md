# Public Protocols

`pyrecest.protocols` is the public home for small capability contracts used by
PyRecEst components.

A protocol describes what an object can do without requiring it to inherit from
a specific base class. This keeps extension points lightweight: a user-defined
class can work with generic PyRecEst utilities when it implements the required
methods and attributes.

## Current scope

The protocol package is intentionally introduced incrementally. The common module
currently defines dimension protocols and broad array aliases:

- `SupportsDim` for objects with an intrinsic state-space dimension;
- `SupportsInputDim` for objects with an ambient or input coordinate dimension;
- `ArrayLike` and `BackendArray` as intentionally broad aliases for backend
  compatible values.

The conversion module defines protocols for representation conversion and
approximation:

- `SupportsFromDistribution` for target classes exposing
  `from_distribution(distribution, ...)`;
- `SupportsConvertTo` for objects exposing `convert_to(...)`;
- `SupportsApproximateAs` for objects exposing `approximate_as(...)`;
- `SupportsDistributionConversion` for objects exposing both conversion
  convenience wrappers;
- `DistributionConverter` for callables registered as conversion routes;
- `ConversionAliasResolver` for callables that resolve aliases to target
  classes.

Follow-up pull requests can add distribution, filter, model, and
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
from pyrecest.protocols.conversions import SupportsFromDistribution
```

Package-level exports are intentionally minimal in this seed package to reduce
merge conflicts while follow-up protocol modules are developed in parallel.
