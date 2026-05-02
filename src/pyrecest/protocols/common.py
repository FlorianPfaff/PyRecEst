"""Common public protocols and aliases for PyRecEst components."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

ArrayLike = Any
"""Array-like input accepted by PyRecEst capability protocols.

This alias is intentionally broad because PyRecEst supports multiple numerical
backends. Concrete implementations should follow the active backend conventions
when backend portability is required.
"""

BackendArray = Any
"""Array object produced by the active PyRecEst backend."""


@runtime_checkable
class SupportsDim(Protocol):
    """Object exposing an intrinsic state-space dimension.

    The dimension refers to the number of independent coordinates used by the
    state representation. For Euclidean vectors this typically matches the
    vector length. For manifold-valued objects it may differ from the embedded
    coordinate dimension.
    """

    @property
    def dim(self) -> int:
        """Intrinsic state-space dimension."""
        raise NotImplementedError


@runtime_checkable
class SupportsInputDim(Protocol):
    """Object exposing an ambient/input coordinate dimension.

    The input dimension describes the trailing coordinate dimension expected by
    public methods such as density evaluation or sampling helpers. For embedded
    manifolds, this may be larger than :attr:`dim`.
    """

    @property
    def input_dim(self) -> int:
        """Ambient/input coordinate dimension."""
        raise NotImplementedError
