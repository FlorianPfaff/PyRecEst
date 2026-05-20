"""Likelihood- and sampling-based reusable model objects.

These classes provide generic, filter-independent containers for model
capabilities commonly needed by particle filters, grid filters, and other
likelihood-based estimators.

The conventions are intentionally simple:

- measurement likelihoods use ``likelihood(measurement, state)``;
- transition samplers use ``sample_next(state, n=1)``;
- transition densities use ``transition_density(state_next, state_previous)``.

The classes do not modify filters directly. They only make the model
capabilities explicit so filter adapters can consume them later without
duplicating callback conventions. The public capability protocols live in
:mod:`pyrecest.protocols.models` and are re-exported here for backwards
compatibility.
"""

from __future__ import annotations

from inspect import Parameter, signature
from typing import Any, Callable

from pyrecest.protocols.models import (
    SupportsLikelihood,
    SupportsLogLikelihood,
    SupportsTransitionDensity,
    SupportsTransitionSampling,
)


def _ensure_callable(value: Any, name: str) -> None:
    if not callable(value):
        raise TypeError(f"{name} must be callable")


def _sample_count_call_mode(callback: Callable[..., Any]) -> str | None:
    """Return how a sampler callback appears to accept an ``n`` argument."""

    try:
        parameters = tuple(signature(callback).parameters.values())
    except (TypeError, ValueError):
        return "positional"

    if any(parameter.kind is Parameter.VAR_POSITIONAL for parameter in parameters):
        return "positional"

    positional_parameters = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional_parameters) >= 2:
        return "positional"

    for parameter in parameters:
        if parameter.name == "n":
            if parameter.kind is Parameter.KEYWORD_ONLY:
                return "keyword"
            return "positional"

    if any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters):
        return "keyword"

    return None


def _evaluate_distribution_method(
    distribution: Any, method_name: str, *args: Any
) -> Any:
    method = getattr(distribution, method_name, None)
    if method is None or not callable(method):
        raise AttributeError(
            f"Distribution object of type {type(distribution).__name__!r} "
            f"does not provide callable method {method_name!r}."
        )
    return method(*args)


class LikelihoodMeasurementModel:
    """Measurement model specified by a likelihood callback."""

    def __init__(
        self,
        likelihood: Callable[[Any, Any], Any],
        *,
        log_likelihood: Callable[[Any, Any], Any] | None = None,
        name: str | None = None,
    ):
        _ensure_callable(likelihood, "likelihood")
        if log_likelihood is not None:
            _ensure_callable(log_likelihood, "log_likelihood")

        self._likelihood = likelihood
        self._log_likelihood = log_likelihood
        self.name = name

    @classmethod
    def from_distribution_factory(
        cls,
        distribution_factory: Callable[[Any], Any],
        *,
        pdf_method: str = "pdf",
        log_pdf_method: str | None = None,
        name: str | None = None,
    ) -> "LikelihoodMeasurementModel":
        """Create a likelihood model from a state-conditioned distribution."""

        _ensure_callable(distribution_factory, "distribution_factory")

        def likelihood(measurement: Any, state: Any) -> Any:
            distribution = distribution_factory(state)
            return _evaluate_distribution_method(distribution, pdf_method, measurement)

        if log_pdf_method is None:
            log_likelihood = None
        else:

            def log_likelihood(measurement: Any, state: Any) -> Any:
                distribution = distribution_factory(state)
                return _evaluate_distribution_method(
                    distribution, log_pdf_method, measurement
                )

        return cls(likelihood, log_likelihood=log_likelihood, name=name)

    @property
    def has_log_likelihood(self) -> bool:
        """Whether this model has an explicit log-likelihood callback."""

        return self._log_likelihood is not None

    def likelihood(self, measurement: Any, state: Any) -> Any:
        """Return values proportional to ``p(measurement | state)``."""

        return self._likelihood(measurement, state)

    def log_likelihood(self, measurement: Any, state: Any) -> Any:
        """Return ``log p(measurement | state)`` if available."""

        if self._log_likelihood is None:
            raise NotImplementedError("No log_likelihood callback was provided.")
        return self._log_likelihood(measurement, state)


class SampleableTransitionModel:
    """Transition model specified by a next-state sampler."""

    def __init__(
        self,
        sample_next: Callable[..., Any],
        *,
        transition_density: Callable[[Any, Any], Any] | None = None,
        name: str | None = None,
        function_is_vectorized: bool = True,
    ):
        _ensure_callable(sample_next, "sample_next")
        if transition_density is not None:
            _ensure_callable(transition_density, "transition_density")

        self._sample_next = sample_next
        self._sample_next_count_call_mode = _sample_count_call_mode(sample_next)
        self._transition_density = transition_density
        self.function_is_vectorized = function_is_vectorized
        self.name = name

    @property
    def has_transition_density(self) -> bool:
        """Whether this model also has a transition-density callback."""

        return self._transition_density is not None

    def sample_next(self, state: Any, n: int = 1) -> Any:
        """Draw ``n`` next-state samples conditioned on ``state``."""

        if self._sample_next_count_call_mode == "positional":
            return self._sample_next(state, n)
        if self._sample_next_count_call_mode == "keyword":
            return self._sample_next(state, n=n)
        return self._sample_next(state)

    def transition_density(self, state_next: Any, state_previous: Any) -> Any:
        """Return transition density values if available."""

        if self._transition_density is None:
            raise NotImplementedError("No transition_density callback was provided.")
        return self._transition_density(state_next, state_previous)


class DensityTransitionModel:
    """Transition model specified by a transition-density callback."""

    def __init__(
        self,
        transition_density: Callable[[Any, Any], Any],
        *,
        sample_next: Callable[[Any, int], Any] | None = None,
        name: str | None = None,
    ):
        _ensure_callable(transition_density, "transition_density")
        if sample_next is not None:
            _ensure_callable(sample_next, "sample_next")

        self._transition_density = transition_density
        self._sample_next = sample_next
        self.name = name

    @property
    def has_sampler(self) -> bool:
        """Whether this model also has a sampler callback."""

        return self._sample_next is not None

    def transition_density(self, state_next: Any, state_previous: Any) -> Any:
        """Return values proportional to ``p(state_next | state_previous)``."""

        return self._transition_density(state_next, state_previous)

    def sample_next(self, state: Any, n: int = 1) -> Any:
        """Draw ``n`` next-state samples if a sampler is available."""

        if self._sample_next is None:
            raise NotImplementedError("No sample_next callback was provided.")
        return self._sample_next(state, n)


__all__ = [
    "DensityTransitionModel",
    "LikelihoodMeasurementModel",
    "SampleableTransitionModel",
    "SupportsLikelihood",
    "SupportsLogLikelihood",
    "SupportsTransitionDensity",
    "SupportsTransitionSampling",
]
