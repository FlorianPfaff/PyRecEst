# Custom Distribution Extensions

A custom distribution does not need to inherit from a PyRecEst base class to be
useful in protocol-oriented code. It only needs to expose the capabilities that a
consumer asks for.

The current public protocol seed provides dimension protocols. Future
distribution-specific protocols can build on the same method names and shape
conventions.

## Minimal dimension contract

Expose `dim` when the object has an intrinsic state-space dimension. Expose
`input_dim` when density evaluation or samples use an ambient coordinate vector
whose length may differ from `dim`.

```python
from pyrecest.protocols.common import SupportsDim, SupportsInputDim


class ScalarDistribution:
    @property
    def dim(self) -> int:
        return 1

    @property
    def input_dim(self) -> int:
        return 1


obj = ScalarDistribution()
assert isinstance(obj, SupportsDim)
assert isinstance(obj, SupportsInputDim)
```

For Euclidean vector states, `dim` and `input_dim` are often the same. For
embedded manifolds, they may differ.

## Distribution-like methods

The examples use method names that already match common PyRecEst distribution
conventions:

- `pdf(xs)` evaluates density values;
- `sample(n)` draws samples;
- `mean()` returns a point estimate or moment-like representative;
- `covariance()` returns a Euclidean covariance when that concept is meaningful.

Do not add placeholder methods only to satisfy a large interface. If an analytic
probability density is unavailable, omit `pdf`. If covariance is not meaningful
on a manifold, omit `covariance` or provide a clearly documented tangent-space
quantity.

## Runnable example

`examples/basic/custom_distribution_protocol.py` implements a small scalar
Gaussian-like class. It is deliberately independent of PyRecEst distribution base
classes:

```python
from pyrecest.protocols.common import SupportsDim, SupportsInputDim


def validate_dimensions(distribution: SupportsDim) -> int:
    return distribution.dim
```

Run it from the repository root:

```bash
python examples/basic/custom_distribution_protocol.py
```

The example demonstrates three points:

- structural protocol checks with `isinstance`;
- a distribution-like class that exposes common PyRecEst method names;
- a generic helper that depends only on `SupportsDim`.

## Checklist for new custom distributions

Use this checklist before introducing a custom distribution to examples, tests,
or generic utilities:

- State whether the object is Euclidean, periodic, spherical, product-space, or
  another manifold-valued representation.
- Document `dim` and, if present, `input_dim`.
- Document accepted input shapes for density evaluation.
- Document sample shape and random-number behavior.
- Prefer backend-compatible arrays when the object should work across NumPy,
  PyTorch, and JAX.
- Avoid claiming support for analytic density, covariance, or closed-form
  products when those operations are only approximate.
