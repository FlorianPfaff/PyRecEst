"""Additive-noise nonlinear transition and measurement models."""

from collections.abc import Callable
from typing import Any

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import asarray


def _as_optional_array(value):
    """Convert ``value`` through the active backend unless it is ``None``."""
    return None if value is None else asarray(value)


def _call_or_value(obj, name):
    """Return an attribute value, calling zero-argument attributes if needed."""
    if obj is None or not hasattr(obj, name):
        return None
    value = getattr(obj, name)
    return value() if callable(value) else value


def _distribution_mean(distribution):
    """Return mean information exposed by a distribution, if any."""
    mean = _call_or_value(distribution, "mean")
    return _call_or_value(distribution, "mu") if mean is None else mean


def _distribution_covariance(distribution):
    """Return covariance information exposed by a distribution, if any."""
    covariance = _call_or_value(distribution, "covariance")
    if covariance is not None:
        return covariance
    covariance = _call_or_value(distribution, "C")
    return _call_or_value(distribution, "cov") if covariance is None else covariance


def _require_callable(function, name):
    if not callable(function):
        raise TypeError(f"{name} must be callable")


class AdditiveNoiseTransitionModel:
    """Nonlinear transition model with additive state noise.

    The model represents ``x_next = f(x) + w`` where ``f`` is the noise-free
    transition function and ``w`` follows the supplied noise distribution. It is
    intentionally filter-independent: sigma-point filters can use
    :meth:`transition_function`, linearized filters can use :meth:`jacobian`, and
    sample- or density-based filters can use :meth:`sample_next` or
    :meth:`transition_density` when the noise distribution supports them.
    """

    def __init__(
        self,
        f: Callable[[Any], Any],
        noise_distribution: Any | None = None,
        noise_mean: Any | None = None,
        noise_covariance: Any | None = None,
        jacobian: Callable[[Any], Any] | None = None,
        vectorized: bool = False,
    ):
        _require_callable(f, "f")
        if jacobian is not None:
            _require_callable(jacobian, "jacobian")
        self._f = f
        self.noise_distribution = noise_distribution
        self._noise_mean = _as_optional_array(noise_mean)
        self._noise_covariance = _as_optional_array(noise_covariance)
        self._jacobian = jacobian
        self.vectorized = vectorized

    def transition_function(self, state):
        """Evaluate the noise-free transition ``f(state)``."""
        return self._f(state)

    def propagate(self, state):
        """Alias for :meth:`transition_function`."""
        return self.transition_function(state)

    @property
    def noise_mean(self):
        """Mean of the additive transition noise, or ``None`` if unavailable."""
        return self._noise_mean if self._noise_mean is not None else _distribution_mean(self.noise_distribution)

    @property
    def noise_covariance(self):
        """Covariance of the additive transition noise, or ``None`` if unavailable."""
        return self._noise_covariance if self._noise_covariance is not None else _distribution_covariance(self.noise_distribution)

    def mean(self, state):
        """Return ``f(state)`` plus the additive noise mean if available."""
        propagated = self.transition_function(state)
        noise_mean = self.noise_mean
        return propagated if noise_mean is None else propagated + noise_mean

    def jacobian(self, state):
        """Return the transition Jacobian evaluated at ``state``."""
        if self._jacobian is None:
            raise NotImplementedError("No transition Jacobian callback was supplied")
        return self._jacobian(state)

    def has_jacobian(self):
        """Return whether this model can provide transition Jacobians."""
        return self._jacobian is not None

    def sample_next(self, state, n: int = 1):
        """Draw ``n`` samples from ``p(x_next | state)``."""
        if self.noise_distribution is None or not hasattr(self.noise_distribution, "sample"):
            raise NotImplementedError("The transition noise distribution does not provide sample(n)")
        return self.transition_function(state) + self.noise_distribution.sample(n)

    def transition_density(self, next_state, state):
        """Evaluate ``p(next_state | state)`` from the additive noise density."""
        if self.noise_distribution is None or not hasattr(self.noise_distribution, "pdf"):
            raise NotImplementedError("The transition noise distribution does not provide pdf(x)")
        return self.noise_distribution.pdf(next_state - self.transition_function(state))


class AdditiveNoiseMeasurementModel:
    """Nonlinear measurement model with additive measurement noise.

    The model represents ``z = h(x) + v`` where ``h`` is the noise-free
    measurement function and ``v`` follows the supplied noise distribution.
    """

    def __init__(
        self,
        h: Callable[[Any], Any],
        noise_distribution: Any | None = None,
        noise_mean: Any | None = None,
        noise_covariance: Any | None = None,
        jacobian: Callable[[Any], Any] | None = None,
        vectorized: bool = False,
    ):
        _require_callable(h, "h")
        if jacobian is not None:
            _require_callable(jacobian, "jacobian")
        self._h = h
        self.noise_distribution = noise_distribution
        self._noise_mean = _as_optional_array(noise_mean)
        self._noise_covariance = _as_optional_array(noise_covariance)
        self._jacobian = jacobian
        self.vectorized = vectorized

    def measurement_function(self, state):
        """Evaluate the noise-free measurement ``h(state)``."""
        return self._h(state)

    def predict_measurement(self, state):
        """Return ``h(state)`` plus the additive noise mean if available."""
        prediction = self.measurement_function(state)
        noise_mean = self.noise_mean
        return prediction if noise_mean is None else prediction + noise_mean

    def mean(self, state):
        """Alias for :meth:`predict_measurement`."""
        return self.predict_measurement(state)

    @property
    def noise_mean(self):
        """Mean of the additive measurement noise, or ``None`` if unavailable."""
        return self._noise_mean if self._noise_mean is not None else _distribution_mean(self.noise_distribution)

    @property
    def noise_covariance(self):
        """Covariance of the additive measurement noise, or ``None`` if unavailable."""
        return self._noise_covariance if self._noise_covariance is not None else _distribution_covariance(self.noise_distribution)

    def jacobian(self, state):
        """Return the measurement Jacobian evaluated at ``state``."""
        if self._jacobian is None:
            raise NotImplementedError("No measurement Jacobian callback was supplied")
        return self._jacobian(state)

    def has_jacobian(self):
        """Return whether this model can provide measurement Jacobians."""
        return self._jacobian is not None

    def measurement_residual(self, measurement, state):
        """Return ``measurement - h(state)``."""
        return measurement - self.measurement_function(state)

    def sample_measurement(self, state, n: int = 1):
        """Draw ``n`` samples from ``p(measurement | state)``."""
        if self.noise_distribution is None or not hasattr(self.noise_distribution, "sample"):
            raise NotImplementedError("The measurement noise distribution does not provide sample(n)")
        return self.measurement_function(state) + self.noise_distribution.sample(n)

    def likelihood(self, measurement, state):
        """Evaluate ``p(measurement | state)`` from the additive noise density."""
        if self.noise_distribution is None or not hasattr(self.noise_distribution, "pdf"):
            raise NotImplementedError("The measurement noise distribution does not provide pdf(x)")
        return self.noise_distribution.pdf(self.measurement_residual(measurement, state))
