# Public Protocols

`pyrecest.protocols` is the public home for small capability contracts used by
PyRecEst components.

A protocol describes what an object can do without requiring it to inherit from
a specific base class. This keeps extension points lightweight: a user-defined
class can work with generic PyRecEst utilities when it implements the required
methods and attributes.

## Current scope

The seed package defines common dimension protocols and broad array aliases:

- `SupportsDim` for objects with an intrinsic state-space dimension;
- `SupportsInputDim` for objects with an ambient or input coordinate dimension;
- `ArrayLike` and `BackendArray` as intentionally broad aliases for backend
  compatible values.

The distribution protocol module adds capability contracts for common
distribution operations such as density evaluation, log-density evaluation,
sampling, moments, modes, multiplication, and convolution.

Follow-up pull requests can add filter, model, conversion, and
manifold-specific protocols independently.

## Design principles

Protocols should stay small and capability-oriented. Instead of requiring every
distribution, model, or filter to implement one large interface, PyRecEst should
ask only for the capability that a function actually needs.

For example, a density utility may require a `SupportsPdf` protocol while a
sampler utility may require only `SupportsSampling`. A particle representation
should not need to implement analytic density evaluation merely to satisfy a
large distribution base interface.

## Distribution protocols

Distribution protocols live in `pyrecest.protocols.distributions` and are
structural. A class does not need to inherit from a PyRecEst abstract base class
to satisfy them.

Available distribution capabilities include:

- `SupportsPdf` for objects with `pdf(xs)`;
- `SupportsLnPdf` for objects with PyRecEst's `ln_pdf(xs)` log-density name;
- `SupportsSampling` for objects with `sample(n)`;
- `SupportsMean` for objects with `mean()`;
- `SupportsCovariance` for objects with `covariance()`;
- `SupportsMode` for objects with `mode(...)`;
- `SupportsModeSetting` for objects with `set_mode(mode)`;
- `SupportsMultiplication` for objects with `multiply(other)`;
- `SupportsConvolution` for objects with `convolve(other)`.

Composite convenience protocols include:

- `DensityLike`, requiring `dim` and `pdf(xs)`;
- `LogDensityLike`, requiring `dim` and `ln_pdf(xs)`;
- `ManifoldDensityLike`, requiring `dim`, `input_dim`, `pdf(xs)`, and
  `mean()`.

Example:

```python
from pyrecest.protocols.distributions import DensityLike, SupportsSampling


def evaluate_density(distribution: DensityLike, xs):
    return distribution.pdf(xs)


def draw_samples(distribution: SupportsSampling, n: int):
    return distribution.sample(n)
```

These protocols intentionally do not define one mandatory distribution
super-interface. Analytic densities, Dirac representations, grid distributions,
mixtures, and manifold-valued distributions can expose different capabilities.

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
from pyrecest.protocols.distributions import SupportsPdf, SupportsSampling
```

Package-level exports are intentionally minimal in this seed package to reduce
merge conflicts while follow-up protocol modules are developed in parallel.
