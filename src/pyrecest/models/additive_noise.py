"""Additive-noise nonlinear transition and measurement models."""

import inspect
from collections.abc import Callable
from typing import Any

# pylint: disable=no-name-in-module,no-member,too-many-instance-attributes,too-many-positional-arguments
from pyrecest.backend import asarray, is_array


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
    if covariance is not None:
        return covariance
    if is_array(distribution):
        return asarray(distribution)
    covariance = _call_or_value(distribution, "cov")
    if covariance is not None:
        return covariance
    if (
        distribution is not None
        and not hasattr(distribution, "sample")
        and not hasattr(distribution, "pdf")
    ):
        return asarray(distribution)
    return None


def _require_callable(function: Any, name: str) -> Callable[..., Any]:
    if not callable(function):
        raise TypeError(f"{name} must be callable")
    return function


def _pop_function_alias(
    kwargs: dict[str, Any],
    canonical_name: str,
    alias_name: str,
    current_value: Any,
) -> Any:
    """Resolve legacy constructor aliases without accepting ambiguous input."""
    alias_value = kwargs.pop(alias_name, None)
    if current_value is not None and alias_value is not None:
        raise TypeError(f"Got both {canonical_name} and {alias_name}")
    return current_value if current_value is not None else alias_value


def _reject_unexpected_kwargs(kwargs: dict[str, Any]) -> None:
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")


def _dt_call_mode(function: Callable[..., Any]) -> str | None:
    """Return how ``dt`` can be passed to ``function``."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return "positional"

    parameters = tuple(signature.parameters.values())
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
        return "keyword"
    if "dt" in signature.parameters:
        return "keyword"
    if any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in parameters):
        return "positional"

    positional = tuple(
        param
        for param in parameters
        if param.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    )
    return "positional" if len(positional) >= 2 else None


def _call_transition_function(
    function: Callable[..., Any],
    state: Any,
    dt: Any,
    function_args: dict[str, Any],
    kwargs: dict[str, Any],
) -> Any:
    """Call a transition function with default and per-call arguments."""
    call_kwargs = {**function_args, **kwargs}
    if dt is None:
        return function(state, **call_kwargs)

    dt_mode = _dt_call_mode(function)
    if dt_mode == "keyword" and "dt" not in call_kwargs:
        return function(state, dt=dt, **call_kwargs)
    if dt_mode == "positional":
        return function(state, dt, **call_kwargs)
    return function(state, **call_kwargs)


def _call_measurement_function(
    function: Callable[..., Any],
    state: Any,
    function_args: dict[str, Any],
    kwargs: dict[str, Any],
) -> Any:
    """Call a measurement function with default and per-call arguments."""
    return function(state, **{**function_args, **kwargs})


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
        f: Callable[..., Any] | None = None,
        noise_distribution: Any | None = None,
        noise_mean=None,
        noise_covariance=None,
        jacobian: Callable[..., Any] | None = None,
        vectorized: bool = False,
        dt=None,
        function_args: dict[str, Any] | None = None,
        **kwargs,
    ):
        f = _pop_function_alias(kwargs, "f", "transition_function", f)
        _reject_unexpected_kwargs(kwargs)

        self._f = _require_callable(f, "f")
        if jacobian is not None:
            jacobian = _require_callable(jacobian, "jacobian")
        self.noise_distribution = noise_distribution
        self._noise_mean = _as_optional_array(noise_mean)
        self._noise_covariance = _as_optional_array(noise_covariance)
        self._jacobian = jacobian
        self.vectorized = vectorized
        self.dt = dt
        self.function_args = dict(function_args or {})

    def evaluate(self, state, dt=None, **kwargs):
        """Evaluate the noise-free transition, including default function args."""
        effective_dt = self.dt if dt is None else dt
        return _call_transition_function(
            self._f, state, effective_dt, self.function_args, kwargs
        )

    def transition_function(self, state, **kwargs):
        """Evaluate the noise-free transition ``f(state)``."""
        return self.evaluate(state, **kwargs)

    def propagate(self, state):
        """Alias for :meth:`transition_function`."""
        return self.transition_function(state)

    @property
    def noise_mean(self):
        """Mean of the additive transition noise, or ``None`` if unavailable."""
        return (
            self._noise_mean
            if self._noise_mean is not None
            else _distribution_mean(self.noise_distribution)
        )

    @property
    def noise_covariance(self):
        """Covariance of the additive transition noise, or ``None`` if unavailable."""
        return (
            self._noise_covariance
            if self._noise_covariance is not None
            else _distribution_covariance(self.noise_distribution)
        )

    def mean(self, state):
        """Return ``f(state)`` plus the additive noise mean if available."""
        propagated = self.transition_function(state)
        noise_mean = self.noise_mean
        return propagated if noise_mean is None else propagated + noise_mean

    def jacobian(self, state):
        """Return the transition Jacobian evaluated at ``state``."""
        jacobian = self._jacobian
        if jacobian is None:
            raise NotImplementedError("No transition Jacobian callback was supplied")
        return jacobian(state)

    def has_jacobian(self):
        """Return whether this model can provide transition Jacobians."""
        return self._jacobian is not None

    def sample_next(self, state, n: int = 1):
        """Draw ``n`` samples from ``p(x_next | state)``."""
        if self.noise_distribution is None or not hasattr(
            self.noise_distribution, "sample"
        ):
            raise NotImplementedError(
                "The transition noise distribution does not provide sample(n)"
            )
        return self.transition_function(state) + self.noise_distribution.sample(n)

    def transition_density(self, next_state, state):
        """Evaluate ``p(next_state | state)`` from the additive noise density."""
        if self.noise_distribution is None or not hasattr(
            self.noise_distribution, "pdf"
        ):
            raise NotImplementedError(
                "The transition noise distribution does not provide pdf(x)"
            )
        return self.noise_distribution.pdf(next_state - self.transition_function(state))


class AdditiveNoiseMeasurementModel:
    """Nonlinear measurement model with additive measurement noise.

    The model represents ``z = h(x) + v`` where ``h`` is the noise-free
    measurement function and ``v`` follows the supplied noise distribution.
    """

    def __init__(
        self,
        h: Callable[..., Any] | None = None,
        noise_distribution: Any | None = None,
        noise_mean=None,
        noise_covariance=None,
        jacobian: Callable[..., Any] | None = None,
        vectorized: bool = False,
        function_args: dict[str, Any] | None = None,
        **kwargs,
    ):
        h = _pop_function_alias(kwargs, "h", "measurement_function", h)
        _reject_unexpected_kwargs(kwargs)

        self._h = _require_callable(h, "h")
        if jacobian is not None:
            jacobian = _require_callable(jacobian, "jacobian")
        self.noise_distribution = noise_distribution
        self._noise_mean = _as_optional_array(noise_mean)
        self._noise_covariance = _as_optional_array(noise_covariance)
        self._jacobian = jacobian
        self.vectorized = vectorized
        self.function_args = dict(function_args or {})

    def evaluate(self, state, **kwargs):
        """Evaluate the noise-free measurement, including default function args."""
        return _call_measurement_function(self._h, state, self.function_args, kwargs)

    def measurement_function(self, state, **kwargs):
        """Evaluate the noise-free measurement ``h(state)``."""
        return self.evaluate(state, **kwargs)

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
        return (
            self._noise_mean
            if self._noise_mean is not None
            else _distribution_mean(self.noise_distribution)
        )

    @property
    def noise_covariance(self):
        """Covariance of the additive measurement noise, or ``None`` if unavailable."""
        return (
            self._noise_covariance
            if self._noise_covariance is not None
            else _distribution_covariance(self.noise_distribution)
        )

    def jacobian(self, state):
        """Return the measurement Jacobian evaluated at ``state``."""
        jacobian = self._jacobian
        if jacobian is None:
            raise NotImplementedError("No measurement Jacobian callback was supplied")
        return jacobian(state)

    def has_jacobian(self):
        """Return whether this model can provide measurement Jacobians."""
        return self._jacobian is not None

    def measurement_residual(self, measurement, state):
        """Return ``measurement - h(state)``."""
        return measurement - self.measurement_function(state)

    def sample_measurement(self, state, n: int = 1):
        """Draw ``n`` samples from ``p(measurement | state)``."""
        if self.noise_distribution is None or not hasattr(
            self.noise_distribution, "sample"
        ):
            raise NotImplementedError(
                "The measurement noise distribution does not provide sample(n)"
            )
        return self.measurement_function(state) + self.noise_distribution.sample(n)

    def likelihood(self, measurement, state):
        """Evaluate ``p(measurement | state)`` from the additive noise density."""
        if self.noise_distribution is None or not hasattr(
            self.noise_distribution, "pdf"
        ):
            raise NotImplementedError(
                "The measurement noise distribution does not provide pdf(x)"
            )
        return self.noise_distribution.pdf(
            self.measurement_residual(measurement, state)
        )
