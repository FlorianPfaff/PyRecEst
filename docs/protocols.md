# Public Protocols

`pyrecest.protocols` is the public home for small capability contracts used by
PyRecEst components.

A protocol describes what an object can do without requiring it to inherit from
a specific base class. This keeps extension points lightweight: a user-defined
class can work with generic PyRecEst utilities when it implements the required
methods and attributes.

## Current scope

The package defines common dimension protocols and broad array aliases:

- `SupportsDim` for objects with an intrinsic state-space dimension;
- `SupportsInputDim` for objects with an ambient or input coordinate dimension;
- `ArrayLike` and `BackendArray` as intentionally broad aliases for backend
  compatible values.

It also defines distribution and filter capability protocols:

- `SupportsPdf` and `SupportsLogPdf` for density evaluation;
- `SupportsSampling` for sample generation;
- `SupportsMean`, `SupportsCovariance`, and `SupportsMeanAndCovariance` for
  moment-style summaries;
- `SupportsDistributionConversion` and `SupportsDistributionApproximation` for
  representation conversion entry points;
- `SupportsFilterState`, `SupportsPointEstimate`, `SupportsLinearPrediction`,
  `SupportsLinearUpdate`, and `SupportsHistoryRecording` for common filter
  capabilities.

## Design principles

Protocols should stay small and capability-oriented. Instead of requiring every
distribution, model, or filter to implement one large interface, PyRecEst should
ask only for the capability that a function actually needs.

For example, a density utility can require `SupportsPdf` while a sampler utility
can require only `SupportsSampling`. A particle representation should not need
to implement analytic density evaluation merely to satisfy a large distribution
base interface.

## Runtime checks

The public protocols are runtime-checkable where practical:

```python
from pyrecest.protocols import SupportsDim, SupportsPdf


class DemoDistribution:
    dim = 2

    def pdf(self, xs):
        return 1.0


assert isinstance(DemoDistribution(), SupportsDim)
assert isinstance(DemoDistribution(), SupportsPdf)
```

Runtime checks confirm that the required attributes or methods are present. They
do not prove mathematical correctness. Protocol-specific tests should check
shapes, backend behavior, and semantics separately.

## Import style

Use package-level imports for public protocols:

```python
from pyrecest.protocols import SupportsDim, SupportsPdf, SupportsSampling
```

Submodule imports remain available when a smaller namespace is preferred:

```python
from pyrecest.protocols.distributions import SupportsLogPdf
```
