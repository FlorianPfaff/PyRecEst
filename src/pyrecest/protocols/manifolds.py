"""Public manifold and state-space capability protocols."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from .common import ArrayLike, BackendArray, SupportsDim, SupportsInputDim

__all__ = [
    "EmbeddedManifoldLike",
    "FiniteMeasureManifoldLike",
    "ManifoldLike",
    "StateSpaceLike",
    "SupportsAngularError",
    "SupportsCoordinateNormalization",
    "SupportsDistance",
    "SupportsDomainFunctionIntegration",
    "SupportsDomainIntegration",
    "SupportsHypersphericalCoordinateConversion",
    "SupportsIntegrationBoundaries",
    "SupportsManifoldSize",
]


@runtime_checkable
class SupportsManifoldSize(Protocol):
    """Object exposing the total measure of a finite-measure state space.

    This mirrors PyRecEst manifold-specific distribution classes that implement
    ``get_manifold_size()`` and ``get_ln_manifold_size()`` for compact or
    otherwise bounded domains. The size is the measure with respect to the
    natural base measure used by the corresponding density.
    """

    def get_manifold_size(self) -> BackendArray:
        """Return the total measure of the state manifold or domain."""
        raise NotImplementedError

    def get_ln_manifold_size(self) -> BackendArray:
        """Return the logarithm of :meth:`get_manifold_size`."""
        raise NotImplementedError


@runtime_checkable
class SupportsDistance(Protocol):
    """Object exposing a distance on its state space."""

    def distance(self, x: ArrayLike, y: ArrayLike) -> BackendArray:
        """Return the distance between two states or batches of states."""
        raise NotImplementedError


@runtime_checkable
class SupportsCoordinateNormalization(Protocol):
    """Object that can project or wrap states into canonical coordinates."""

    def normalize(self, x: ArrayLike) -> BackendArray:
        """Return ``x`` represented in canonical coordinates for the state space."""
        raise NotImplementedError


@runtime_checkable
class SupportsAngularError(Protocol):
    """Object exposing a periodic angular-error operation."""

    def angular_error(self, alpha: ArrayLike, beta: ArrayLike) -> BackendArray:
        """Return the wrapped angular error between two angular states."""
        raise NotImplementedError


@runtime_checkable
class SupportsDomainIntegration(Protocol):
    """Object that can integrate its represented density over a domain."""

    def integrate(self, integration_boundaries: ArrayLike | None = None) -> BackendArray:
        """Integrate over the state space or an explicitly supplied subdomain."""
        raise NotImplementedError


@runtime_checkable
class SupportsDomainFunctionIntegration(Protocol):
    """Object exposing integration of arbitrary functions over its domain."""

    def integrate_fun_over_domain(
        self,
        f: Callable[..., BackendArray],
        dim: int,
    ) -> BackendArray:
        """Integrate ``f`` over the full domain for the given intrinsic dimension."""
        raise NotImplementedError


@runtime_checkable
class SupportsIntegrationBoundaries(Protocol):
    """Object exposing full-domain integration boundaries."""

    def get_full_integration_boundaries(self, dim: int) -> BackendArray:
        """Return coordinate boundaries for the full integration domain."""
        raise NotImplementedError


@runtime_checkable
class SupportsHypersphericalCoordinateConversion(Protocol):
    """Object exposing hyperspherical/ambient coordinate conversions."""

    def hypersph_to_cart(
        self,
        hypersph_coords: ArrayLike,
        mode: str = "colatitude",
    ) -> BackendArray:
        """Convert hyperspherical coordinates to ambient Cartesian coordinates."""
        raise NotImplementedError

    def cart_to_hypersph(
        self,
        cart_coords: ArrayLike,
        mode: str = "colatitude",
    ) -> BackendArray:
        """Convert ambient Cartesian coordinates to hyperspherical coordinates."""
        raise NotImplementedError


class StateSpaceLike(SupportsDim, SupportsInputDim, Protocol):
    """Minimal protocol for objects tied to a state space.

    ``dim`` is the intrinsic state dimension. ``input_dim`` is the trailing
    coordinate dimension accepted by public methods. These dimensions coincide
    for many Euclidean and toroidal objects, but can differ for embedded
    manifolds such as hyperspheres.
    """


class ManifoldLike(StateSpaceLike, Protocol):
    """Minimal protocol for objects tied to a known manifold.

    The protocol deliberately does not require density evaluation, sampling,
    moments, or finite domain size. Those capabilities should be expressed by
    narrower protocols so manifold metadata remains reusable for distributions,
    filters, models, grids, and user-defined state-space objects.
    """


class EmbeddedManifoldLike(ManifoldLike, Protocol):
    """Marker protocol for manifolds represented in ambient coordinates.

    For hyperspheres, for example, ``input_dim`` is typically ``dim + 1``.
    """


class FiniteMeasureManifoldLike(ManifoldLike, SupportsManifoldSize, Protocol):
    """Protocol for manifolds or domains with a finite total measure."""
