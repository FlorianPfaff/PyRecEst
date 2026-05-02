"""Public model capability protocols for PyRecEst components."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .common import ArrayLike, BackendArray


@runtime_checkable
class SupportsLikelihood(Protocol):
    """Measurement model that can evaluate ``p(measurement | state)``."""

    def likelihood(self, measurement: ArrayLike, state: ArrayLike) -> BackendArray:
        """Return likelihood values for ``measurement`` conditioned on ``state``."""
        raise NotImplementedError


@runtime_checkable
class SupportsLogLikelihood(Protocol):
    """Measurement model that can evaluate log-likelihoods."""

    def log_likelihood(self, measurement: ArrayLike, state: ArrayLike) -> BackendArray:
        """Return log-likelihood values for ``measurement`` conditioned on ``state``."""
        raise NotImplementedError


@runtime_checkable
class SupportsTransitionSampling(Protocol):
    """Transition model that can sample ``p(state_next | state)``."""

    def sample_next(self, state: ArrayLike, n: int = 1) -> BackendArray:
        """Draw ``n`` next-state samples conditioned on ``state``."""
        raise NotImplementedError


@runtime_checkable
class SupportsTransitionDensity(Protocol):
    """Transition model that can evaluate ``p(state_next | state_previous)``."""

    def transition_density(
        self,
        state_next: ArrayLike,
        state_previous: ArrayLike,
    ) -> BackendArray:
        """Return transition-density values."""
        raise NotImplementedError


@runtime_checkable
class SupportsPredictedDistribution(Protocol):
    """Model that can propagate a distribution object directly."""

    def predict_distribution(
        self,
        state_distribution: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Return the predicted distribution for ``state_distribution``."""
        raise NotImplementedError


@runtime_checkable
class SupportsLinearGaussianTransition(Protocol):
    """Canonical structural interface for linear Gaussian transition models."""

    @property
    def system_matrix(self) -> BackendArray:
        """State-transition matrix ``F``."""
        raise NotImplementedError

    @property
    def system_noise_cov(self) -> BackendArray:
        """Process-noise covariance ``Q``."""
        raise NotImplementedError

    @property
    def sys_input(self) -> BackendArray | None:
        """Optional deterministic transition input ``u``."""
        raise NotImplementedError


@runtime_checkable
class SupportsLinearGaussianMeasurement(Protocol):
    """Canonical structural interface for linear Gaussian measurement models."""

    @property
    def measurement_matrix(self) -> BackendArray:
        """Measurement matrix ``H``."""
        raise NotImplementedError

    @property
    def measurement_noise_cov(self) -> BackendArray:
        """Measurement-noise covariance ``R``."""
        raise NotImplementedError


__all__ = [
    "SupportsLikelihood",
    "SupportsLinearGaussianMeasurement",
    "SupportsLinearGaussianTransition",
    "SupportsLogLikelihood",
    "SupportsPredictedDistribution",
    "SupportsTransitionDensity",
    "SupportsTransitionSampling",
]
