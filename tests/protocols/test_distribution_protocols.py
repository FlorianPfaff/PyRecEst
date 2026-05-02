"""Tests for public distribution capability protocols."""

from __future__ import annotations

from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution, VonMisesDistribution
from pyrecest.protocols.common import SupportsDim, SupportsInputDim
from pyrecest.protocols.distributions import (
    DensityLike,
    LogDensityLike,
    ManifoldDensityLike,
    SupportsConvolution,
    SupportsCovariance,
    SupportsLnPdf,
    SupportsMean,
    SupportsMode,
    SupportsModeSetting,
    SupportsMultiplication,
    SupportsPdf,
    SupportsSampling,
)


class MinimalDensity:
    dim = 1

    def pdf(self, xs):
        return xs


class MinimalManifoldDensity:
    dim = 1
    input_dim = 2

    def pdf(self, xs):
        return xs

    def mean(self):
        return array([1.0, 0.0])


class MinimalLogDensity:
    dim = 1

    def ln_pdf(self, xs):
        return xs


def test_distribution_protocols_are_runtime_checkable_for_gaussian():
    gaussian = GaussianDistribution(array([0.0, 0.0]), eye(2))

    assert isinstance(gaussian, SupportsDim)
    assert isinstance(gaussian, SupportsInputDim)
    assert isinstance(gaussian, SupportsPdf)
    assert isinstance(gaussian, SupportsLnPdf)
    assert isinstance(gaussian, SupportsSampling)
    assert isinstance(gaussian, SupportsMean)
    assert isinstance(gaussian, SupportsCovariance)
    assert isinstance(gaussian, SupportsMode)
    assert isinstance(gaussian, SupportsModeSetting)
    assert isinstance(gaussian, SupportsMultiplication)
    assert isinstance(gaussian, SupportsConvolution)
    assert isinstance(gaussian, DensityLike)
    assert isinstance(gaussian, LogDensityLike)
    assert isinstance(gaussian, ManifoldDensityLike)


def test_distribution_protocols_are_runtime_checkable_for_circular_distribution():
    von_mises = VonMisesDistribution(0.0, 1.0)

    assert isinstance(von_mises, SupportsDim)
    assert isinstance(von_mises, SupportsInputDim)
    assert isinstance(von_mises, SupportsPdf)
    assert isinstance(von_mises, SupportsLnPdf)
    assert isinstance(von_mises, SupportsSampling)
    assert isinstance(von_mises, SupportsMean)
    assert isinstance(von_mises, SupportsMultiplication)
    assert isinstance(von_mises, SupportsConvolution)
    assert isinstance(von_mises, DensityLike)
    assert isinstance(von_mises, LogDensityLike)
    assert isinstance(von_mises, ManifoldDensityLike)


def test_density_like_accepts_structural_implementations_without_inheritance():
    density = MinimalDensity()

    assert isinstance(density, SupportsPdf)
    assert isinstance(density, DensityLike)


def test_log_density_like_accepts_structural_implementations_without_inheritance():
    density = MinimalLogDensity()

    assert isinstance(density, SupportsLnPdf)
    assert isinstance(density, LogDensityLike)


def test_manifold_density_like_accepts_structural_implementations_without_inheritance():
    density = MinimalManifoldDensity()

    assert isinstance(density, SupportsInputDim)
    assert isinstance(density, SupportsPdf)
    assert isinstance(density, SupportsMean)
    assert isinstance(density, ManifoldDensityLike)


def test_sampling_protocol_does_not_imply_density_protocol():
    class SampleOnlyDistribution:
        def sample(self, n):
            return array([0.0] * n)

    distribution = SampleOnlyDistribution()

    assert isinstance(distribution, SupportsSampling)
    assert not isinstance(distribution, SupportsPdf)


def test_pdf_protocol_does_not_imply_sampling_protocol():
    distribution = MinimalDensity()

    assert isinstance(distribution, SupportsPdf)
    assert not isinstance(distribution, SupportsSampling)
