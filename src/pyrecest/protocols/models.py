"""Public model capability protocols for PyRecEst.

These protocols describe reusable transition and measurement model capabilities
without requiring user-defined models to inherit from concrete PyRecEst model
classes. They are intentionally small so filters, adapters, and evaluation code
can request only the capabilities they need.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .common import ArrayLike, BackendArray


@runtime_checkable
class SupportsLikelihood(Protocol):
    """Measurement model that can evaluate likelihood values ``p(z | x)``."""

    def likelihood(self, measurement: ArrayLike, state: ArrayLike) -> BackendArray:
        """Return the likelihood of ``measurement`` conditioned on ``state``."""
        raise NotImplementedError


@runtime_checkable
class SupportsLogLikelihood(Protocol):
    """Measurement model that can evaluate log-likelihood values."""

    def log_likelihood(self, measurement: ArrayLike, state: ArrayLike) -> BackendArray:
        """Return ``log p(measurement | state)``."""
        raise NotImplementedError


@runtime_checkable
class SupportsTransitionSampling(Protocol):
    """Transition model that can sample from ``p(x_k | x_{k-1})``."""

    def sample_next(self, state: ArrayLike, n: int = 1) -> BackendArray:
        """Draw ``n`` next-state samples conditioned on ``state``."""
        raise NotImplementedError


@runtime_checkable
class SupportsTransitionDensity(Protocol):
    """Transition model that can evaluate ``p(x_k | x_{k-1})``."""

    def transition_density(
        self, state_next: ArrayLike, state_previous: ArrayLike
    ) -> BackendArray:
        """Return transition density values."""
        raise NotImplementedError


@runtime_checkable
class SupportsPredictedDistribution(Protocol):
    """Model that can propagate a distribution into a predicted distribution."""

    def predict_distribution(
        self, state_distribution: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """Return the distribution implied by propagating ``state_distribution``."""
        raise NotImplementedError


@runtime_checkable
class SupportsLinearGaussianTransition(Protocol):
    """Linear Gaussian transition model with explicit matrix and noise terms."""

    @property
    def state_dim(self) -> int:
        """Input state dimension."""
        raise NotImplementedError

    @property
    def predicted_dim(self) -> int:
        """Predicted state dimension."""
        raise NotImplementedError

    @property
    def system_matrix(self) -> BackendArray:
        """Linear transition matrix."""
        raise NotImplementedError

    @property
    def system_noise_cov(self) -> BackendArray:
        """Additive system-noise covariance matrix."""
        raise NotImplementedError

    def predict_mean(self, state_mean: ArrayLike) -> BackendArray:
        """Return the predicted mean for ``state_mean``."""
        raise NotImplementedError

    def predict_covariance(self, state_covariance: ArrayLike) -> BackendArray:
        """Return the predicted covariance for ``state_covariance``."""
        raise NotImplementedError

    def predict_distribution(
        self, state_distribution: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """Return the predicted distribution for ``state_distribution``."""
        raise NotImplementedError

    def noise_distribution(self, *args: Any, **kwargs: Any) -> Any:
        """Return the additive system-noise distribution."""
        raise NotImplementedError


@runtime_checkable
class SupportsLinearGaussianMeasurement(Protocol):
    """Linear Gaussian measurement model with explicit matrix and noise terms."""

    @property
    def state_dim(self) -> int:
        """Input state dimension."""
        raise NotImplementedError

    @property
    def measurement_dim(self) -> int:
        """Measurement dimension."""
        raise NotImplementedError

    @property
    def measurement_matrix(self) -> BackendArray:
        """Linear measurement matrix."""
        raise NotImplementedError

    @property
    def measurement_noise_cov(self) -> BackendArray:
        """Additive measurement-noise covariance matrix."""
        raise NotImplementedError

    def predict_mean(self, state_mean: ArrayLike) -> BackendArray:
        """Return the predicted measurement mean for ``state_mean``."""
        raise NotImplementedError

    def innovation_covariance(self, state_covariance: ArrayLike) -> BackendArray:
        """Return the innovation covariance for ``state_covariance``."""
        raise NotImplementedError

    def predict_distribution(
        self, state_distribution: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """Return the predicted measurement distribution for ``state_distribution``."""
        raise NotImplementedError

    def noise_distribution(self, *args: Any, **kwargs: Any) -> Any:
        """Return the additive measurement-noise distribution."""
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
