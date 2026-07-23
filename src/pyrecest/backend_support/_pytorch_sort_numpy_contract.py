"""PyTorch sort helpers for backend compatibility."""

from __future__ import annotations

from operator import index as _operator_index

import numpy as _np

_SORT_KIND_MESSAGE = (
    "sort kind must be one of 'quicksort', 'heapsort', 'stable', or 'mergesort'"
)
_SORT_CONFLICT_MESSAGE = "sort() got both 'kind' and 'stable' arguments"
_ARGSORT_CONFLICT_MESSAGE = "argsort() got conflicting 'kind' and 'stable' arguments"
_ARGSORT_DEFAULT_AXIS = object()


def normalize_sort_axis(axis, torch_module=None):
    """Return a sort axis while preserving NumPy's flatten-all sentinel."""
    if axis is None:
        return None
    if isinstance(axis, (bool, _np.bool_)) or bool(
        torch_module is not None
        and torch_module.is_tensor(axis)
        and axis.ndim == 0
        and axis.dtype == torch_module.bool
    ):
        raise TypeError("an integer is required for the axis")
    return _operator_index(axis)


def resolve_sort_stability(kind, stable):
    """Return the torch ``stable`` flag implied by NumPy-style options."""
    # NumPy treats ``kind`` and ``stable`` as mutually exclusive sort-mode
    # selectors, even when they would imply the same stable/unstable choice.
    if kind is not None and stable is not None:
        raise ValueError(_SORT_CONFLICT_MESSAGE)
    if kind is None:
        return stable
    if stable is not None:
        raise TypeError(_SORT_CONFLICT_MESSAGE)
    if kind in {"stable", "mergesort"}:
        return True
    if kind in {"quicksort", "heapsort"}:
        return False
    raise ValueError(_SORT_KIND_MESSAGE)


def sort_axis_none(
    backend_module,
    torch_module,
    values,
    axis=-1,
    kind=None,
    order=None,
    *,
    stable=None,
    descending=False,
):
    """Sort values with NumPy-style ``axis=None`` and keyword support."""
    if order is not None:
        raise ValueError("order is not supported by this backend")
    values = backend_module.array(values)
    axis = normalize_sort_axis(axis, torch_module)
    stable = resolve_sort_stability(kind, stable)
    if axis is None:
        values = torch_module.flatten(values)
        axis = 0
    return torch_module.sort(
        values,
        dim=axis,
        descending=bool(descending),
        stable=bool(stable) if stable is not None else False,
    ).values


def patch_argsort_stability_contract() -> None:
    """Patch PyTorch argsort to reject simultaneous ``kind`` and ``stable``."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    original_argsort = getattr(raw_pytorch, "argsort", None)
    if original_argsort is None:
        return
    if getattr(original_argsort, "_pyrecest_argsort_stability_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.argsort = original_argsort
        return

    def argsort(
        a,
        axis=_ARGSORT_DEFAULT_AXIS,
        kind=None,
        order=None,
        *,
        stable=None,
        dim=None,
        descending=False,
    ):
        if kind is not None and stable is not None:
            raise ValueError(_ARGSORT_CONFLICT_MESSAGE)
        if axis is _ARGSORT_DEFAULT_AXIS:
            return original_argsort(
                a,
                kind=kind,
                order=order,
                stable=stable,
                dim=dim,
                descending=descending,
            )
        return original_argsort(
            a,
            axis=axis,
            kind=kind,
            order=order,
            stable=stable,
            dim=dim,
            descending=descending,
        )

    argsort.__name__ = getattr(original_argsort, "__name__", "argsort")
    argsort.__doc__ = getattr(original_argsort, "__doc__", None)
    argsort._pyrecest_arraylike_contract = getattr(
        original_argsort,
        "_pyrecest_arraylike_contract",
        True,
    )
    argsort._pyrecest_numpy_contract = getattr(
        original_argsort,
        "_pyrecest_numpy_contract",
        True,
    )
    argsort._pyrecest_argsort_stability_contract = True
    raw_pytorch.argsort = argsort
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.argsort = argsort


patch_argsort_stability_contract()
