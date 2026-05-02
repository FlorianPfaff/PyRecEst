# Custom Filter Extensions

A custom recursive filter can follow PyRecEst conventions without inheriting from
a built-in filter class. The protocol-oriented style is to expose only the
capabilities that generic code needs.

The current public protocol seed provides common dimension protocols. Future
filter-specific protocols can formalize additional capabilities such as
`filter_state`, `predict_linear`, `update_linear`, and `get_point_estimate`.

## Minimal dimension contract

Expose `dim` for the intrinsic state dimension:

```python
from pyrecest.protocols.common import SupportsDim


class MyFilter:
    @property
    def dim(self) -> int:
        return 2


assert isinstance(MyFilter(), SupportsDim)
```

This is enough for utilities that only need to know the state dimension.

## Filter-like methods

Many PyRecEst filters expose a current posterior-like state and a point estimate.
For a custom filter, prefer these names unless there is a strong reason not to:

- `filter_state` for the current posterior or state summary;
- `predict(...)` or a more specific prediction method for time propagation;
- `update(...)` or a more specific update method for conditioning on data;
- `get_point_estimate()` for the current state estimate.

The method signatures can be domain-specific. A scalar demonstration filter does
not need the same update signature as a multi-target tracker.

## Runnable example

`examples/basic/custom_filter_protocol.py` implements a scalar recursive filter
that maintains a Gaussian-like state summary and performs predict/update cycles.
It is deliberately independent of PyRecEst filter base classes:

```python
from pyrecest.protocols.common import SupportsDim


def validate_filter_dimension(filter_obj: SupportsDim) -> int:
    return filter_obj.dim
```

Run it from the repository root:

```bash
python examples/basic/custom_filter_protocol.py
```

The example demonstrates three points:

- structural protocol checks with `isinstance`;
- a filter-like class with `filter_state` and `get_point_estimate()`;
- a generic helper that depends only on `SupportsDim`.

## Checklist for new custom filters

Use this checklist before introducing a custom filter to examples, tests, or
generic utilities:

- Document what `filter_state` contains.
- Document whether `dim` is intrinsic, ambient, or task-specific.
- Separate prediction and update behavior unless the filter has a clear reason
  to combine them.
- Document measurement shapes and state shapes.
- Keep backend behavior explicit when arrays are involved.
- Avoid implementing no-op methods only to satisfy an interface that the filter
  does not actually support.
