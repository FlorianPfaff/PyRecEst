"""Distribution capability protocols."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .common import ArrayLike, BackendArray


@runtime_checkable
class SupportsPdf(Protocol):
    """Object that can evaluate a probability density."""

    def pdf(self, xs: ArrayLike) -> BackendArray:
        """Evaluate the density at one or more points."""
        raise NotImplementedError


@runtime_checkable
class SupportsLogPdf(Protocol):
    """Object that can evaluate a log probability density."""

    def ln_pdf(self, xs: ArrayLike) -> BackendArray:
        """Evaluate the log density at one or more points."""
        raise NotImplementedError


@runtime_checkable
class SupportsSampling(Protocol):
    """Object that can draw samples."""

    def sample(self, n: int) -> BackendArray:
        """Draw ``n`` samples."""
        raise NotImplementedError


@runtime_checkable
class SupportsMean(Protocol):
    """Object exposing a mean estimate."""

    def mean(self) -> BackendArray:
        """Return the mean."""
        raise NotImplementedError


@runtime_checkable
class SupportsCovariance(Protocol):
    """Object exposing a covariance estimate."""

    def covariance(self) -> BackendArray:
        """Return the covariance."""
        raise NotImplementedError


@runtime_checkable
class SupportsMeanAndCovariance(SupportsMean, SupportsCovariance, Protocol):
    """Object exposing both mean and covariance estimates."""


@runtime_checkable
class SupportsDistributionConversion(Protocol):
    """Object that can convert itself to another distribution representation."""

    def convert_to(self, target, **kwargs):
        """Convert to ``target`` representation."""
        raise NotImplementedError


@runtime_checkable
class SupportsDistributionApproximation(Protocol):
    """Object that can approximate itself as another representation."""

    def approximate_as(self, target, **kwargs):
        """Approximate as ``target`` representation."""
        raise NotImplementedError
