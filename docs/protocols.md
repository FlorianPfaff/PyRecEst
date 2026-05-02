# Public Protocols

`pyrecest.protocols` is the public home for small capability contracts used by
PyRecEst components.

A protocol describes what an object can do without requiring it to inherit from
a specific base class. This keeps extension points lightweight: a user-defined
class can work with generic PyRecEst utilities when it implements the required
methods and attributes.

## Current scope

This seed package only defines common dimension protocols and broad array
aliases:

- `SupportsDim` for objects with an intrinsic state-space dimension;
- `SupportsInputDim` for objects with an ambient or input coordinate dimension;
- `ArrayLike` and `BackendArray` as intentionally broad aliases for backend
  compatible values.

Follow-up pull requests can add distribution, filter, model, conversion, and
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

Submodule imports remain the clearest option when code depends on one specific
capability area:

```python
from pyrecest.protocols.common import SupportsDim, SupportsInputDim
```

The package also provides curated convenience exports for all available protocol
submodules:

```python
from pyrecest.protocols import SupportsDim, SupportsInputDim
```

Each protocol submodule owns its public names through its own `__all__`. The
package-level namespace re-exports those names from known protocol submodules
that are present in the installed package. This keeps follow-up protocol pull
requests independent while still allowing a stable package-level import style
once the corresponding submodule has landed.
