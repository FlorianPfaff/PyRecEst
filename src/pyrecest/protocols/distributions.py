"""Public distribution capability protocols for PyRecEst components."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .common import ArrayLike, BackendArray, SupportsDim, SupportsInputDim


@runtime_checkable
class SupportsPdf(Protocol):
    """Object supporting probability density evaluation."""

    def pdf(self, xs: ArrayLike) -> BackendArray:
        """Evaluate density values at ``xs``."""
        raise NotImplementedError


@runtime_checkable
class SupportsLnPdf(Protocol):
    """Object supporting log-density evaluation using PyRecEst's ``ln_pdf`` name."""

    def ln_pdf(self, xs: ArrayLike) -> BackendArray:
        """Evaluate log-density values at ``xs``."""
        raise NotImplementedError


@runtime_checkable
class SupportsSampling(Protocol):
    """Object supporting random sampling."""

    def sample(self, n: Any) -> BackendArray:
        """Draw ``n`` samples."""
        raise NotImplementedError


@runtime_checkable
class SupportsMean(Protocol):
    """Object exposing a mean or manifold-appropriate mean representative."""

    def mean(self) -> BackendArray:
        """Return the mean representative."""
        raise NotImplementedError


@runtime_checkable
class SupportsCovariance(Protocol):
    """Object exposing covariance information."""

    def covariance(self) -> BackendArray:
        """Return the covariance representation."""
        raise NotImplementedError


@runtime_checkable
class SupportsMode(Protocol):
    """Object exposing a mode computation."""

    def mode(self, *args: Any, **kwargs: Any) -> BackendArray:
        """Return a mode of the represented distribution."""
        raise NotImplementedError


@runtime_checkable
class SupportsModeSetting(Protocol):
    """Object supporting construction or mutation with a new mode."""

    def set_mode(self, mode: ArrayLike) -> Any:
        """Set or return a representation with the requested mode."""
        raise NotImplementedError


@runtime_checkable
class SupportsMultiplication(Protocol):
    """Object supporting normalized or representation-specific density products."""

    def multiply(self, other: Any) -> Any:
        """Multiply this distribution with ``other``."""
        raise NotImplementedError


@runtime_checkable
class SupportsConvolution(Protocol):
    """Object supporting convolution with another compatible distribution."""

    def convolve(self, other: Any) -> Any:
        """Convolve this distribution with ``other``."""
        raise NotImplementedError


@runtime_checkable
class DensityLike(SupportsDim, SupportsPdf, Protocol):
    """Minimal density-like object with intrinsic dimension and ``pdf``."""


@runtime_checkable
class LogDensityLike(SupportsDim, SupportsLnPdf, Protocol):
    """Minimal log-density-like object with intrinsic dimension and ``ln_pdf``."""


@runtime_checkable
class ManifoldDensityLike(
    SupportsDim,
    SupportsInputDim,
    SupportsPdf,
    SupportsMean,
    Protocol,
):
    """Minimal manifold-aware density-like object."""


__all__ = [
    "DensityLike",
    "LogDensityLike",
    "ManifoldDensityLike",
    "SupportsConvolution",
    "SupportsCovariance",
    "SupportsLnPdf",
    "SupportsMean",
    "SupportsMode",
    "SupportsModeSetting",
    "SupportsMultiplication",
    "SupportsPdf",
    "SupportsSampling",
]
