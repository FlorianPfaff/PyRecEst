"""Smoke tests for public manifold and state-space protocols."""

from __future__ import annotations

from pyrecest.protocols.manifolds import (
    EmbeddedManifoldLike,
    FiniteMeasureManifoldLike,
    ManifoldLike,
    StateSpaceLike,
    SupportsAngularError,
    SupportsCoordinateNormalization,
    SupportsDistance,
    SupportsDomainFunctionIntegration,
    SupportsDomainIntegration,
    SupportsHypersphericalCoordinateConversion,
    SupportsIntegrationBoundaries,
    SupportsManifoldSize,
)


class MinimalStateSpace:
    dim = 2
    input_dim = 3


class FiniteMeasureStateSpace(MinimalStateSpace):
    def get_manifold_size(self):
        return 4.0

    def get_ln_manifold_size(self):
        return 1.3862943611198906


class DistanceStateSpace:
    def distance(self, x, y):
        return abs(x - y)


class CoordinateNormalizingStateSpace:
    def normalize(self, x):
        return x


class PeriodicStateSpace:
    def angular_error(self, alpha, beta):
        return alpha - beta


class IntegratingStateSpace:
    def integrate(self, integration_boundaries=None):
        del integration_boundaries
        return 1.0

    def integrate_fun_over_domain(self, f, dim):
        return f(*range(dim))

    def get_full_integration_boundaries(self, dim):
        return [(0.0, 1.0)] * dim


class HypersphericalCoordinateConverter:
    def hypersph_to_cart(self, hypersph_coords, mode="colatitude"):
        del mode
        return hypersph_coords

    def cart_to_hypersph(self, cart_coords, mode="colatitude"):
        del mode
        return cart_coords


def test_state_space_protocols_are_runtime_checkable():
    obj = MinimalStateSpace()

    assert isinstance(obj, StateSpaceLike)
    assert isinstance(obj, ManifoldLike)
    assert isinstance(obj, EmbeddedManifoldLike)
    assert not isinstance(obj, FiniteMeasureManifoldLike)


def test_finite_measure_manifold_protocol_is_runtime_checkable():
    obj = FiniteMeasureStateSpace()

    assert isinstance(obj, SupportsManifoldSize)
    assert isinstance(obj, FiniteMeasureManifoldLike)


def test_geometric_operation_protocols_are_runtime_checkable():
    assert isinstance(DistanceStateSpace(), SupportsDistance)
    assert isinstance(CoordinateNormalizingStateSpace(), SupportsCoordinateNormalization)
    assert isinstance(PeriodicStateSpace(), SupportsAngularError)


def test_domain_integration_protocols_are_runtime_checkable():
    obj = IntegratingStateSpace()

    assert isinstance(obj, SupportsDomainIntegration)
    assert isinstance(obj, SupportsDomainFunctionIntegration)
    assert isinstance(obj, SupportsIntegrationBoundaries)


def test_coordinate_conversion_protocol_is_runtime_checkable():
    assert isinstance(
        HypersphericalCoordinateConverter(),
        SupportsHypersphericalCoordinateConversion,
    )
