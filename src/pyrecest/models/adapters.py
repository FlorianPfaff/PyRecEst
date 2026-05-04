"""Protocol-based adapter helpers for reusable model objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pyrecest.protocols.models import (
    SupportsLikelihood,
    SupportsLogLikelihood,
    SupportsPredictedDistribution,
    SupportsTransitionDensity,
    SupportsTransitionSampling,
)

from .likelihood import (
    DensityTransitionModel,
    LikelihoodMeasurementModel,
    SampleableTransitionModel,
)

_MODEL_ATTRIBUTE_SENTINEL = object()


@dataclass(frozen=True)
class LinearTransitionArguments:
    """Arguments consumed by linear Gaussian prediction APIs."""

    system_matrix: Any
    sys_noise_cov: Any
    sys_input: Any | None = None


@dataclass(frozen=True)
class LinearMeasurementArguments:
    """Arguments consumed by linear Gaussian measurement-update APIs."""

    measurement_matrix: Any
    meas_noise: Any


def _require_capability(obj: object, protocol: Any, capability_name: str) -> None:
    if not isinstance(obj, protocol):
        raise TypeError(f"{type(obj).__name__} must support {capability_name}.")


def _metadata_was_requested(*, name: str | None, extra: Any | None = None) -> bool:
    return name is not None or extra is not None


def as_likelihood_model(
    model_or_likelihood: SupportsLikelihood | Callable[[Any, Any], Any],
    *,
    log_likelihood: Callable[[Any, Any], Any] | None = None,
    name: str | None = None,
) -> SupportsLikelihood:
    """Return a likelihood-capable model, wrapping callbacks when needed."""

    if isinstance(model_or_likelihood, SupportsLikelihood):
        if _metadata_was_requested(name=name, extra=log_likelihood):
            raise ValueError(
                "log_likelihood and name are only used when wrapping a callable."
            )
        return model_or_likelihood

    if callable(model_or_likelihood):
        return LikelihoodMeasurementModel(
            model_or_likelihood,
            log_likelihood=log_likelihood,
            name=name,
        )

    raise TypeError("model_or_likelihood must be a likelihood model or callable")


def as_sampleable_transition_model(
    model_or_sampler: SupportsTransitionSampling | Callable[[Any, int], Any],
    *,
    transition_density: Callable[[Any, Any], Any] | None = None,
    name: str | None = None,
) -> SupportsTransitionSampling:
    """Return a sampleable transition model, wrapping callbacks when needed."""

    if isinstance(model_or_sampler, SupportsTransitionSampling):
        if _metadata_was_requested(name=name, extra=transition_density):
            raise ValueError(
                "transition_density and name are only used when wrapping a callable."
            )
        return model_or_sampler

    if callable(model_or_sampler):
        return SampleableTransitionModel(
            model_or_sampler,
            transition_density=transition_density,
            name=name,
        )

    raise TypeError("model_or_sampler must be a transition model or callable")


def as_density_transition_model(
    model_or_density: SupportsTransitionDensity | Callable[[Any, Any], Any],
    *,
    sample_next: Callable[[Any, int], Any] | None = None,
    name: str | None = None,
) -> SupportsTransitionDensity:
    """Return a density-capable transition model, wrapping callbacks when needed."""

    if isinstance(model_or_density, SupportsTransitionDensity):
        if _metadata_was_requested(name=name, extra=sample_next):
            raise ValueError(
                "sample_next and name are only used when wrapping a callable."
            )
        return model_or_density

    if callable(model_or_density):
        return DensityTransitionModel(
            model_or_density,
            sample_next=sample_next,
            name=name,
        )

    raise TypeError("model_or_density must be a transition-density model or callable")


def evaluate_likelihood(model: SupportsLikelihood, measurement: Any, state: Any) -> Any:
    """Evaluate ``p(measurement | state)`` on a likelihood-capable model."""

    _require_capability(model, SupportsLikelihood, "likelihood(measurement, state)")
    return model.likelihood(measurement, state)


def evaluate_log_likelihood(
    model: SupportsLogLikelihood,
    measurement: Any,
    state: Any,
) -> Any:
    """Evaluate ``log p(measurement | state)`` on a compatible model."""

    _require_capability(
        model, SupportsLogLikelihood, "log_likelihood(measurement, state)"
    )
    return model.log_likelihood(measurement, state)


def sample_next_state(model: SupportsTransitionSampling, state: Any, n: int = 1) -> Any:
    """Draw next-state samples from a sampleable transition model."""

    _require_capability(model, SupportsTransitionSampling, "sample_next(state, n)")
    return model.sample_next(state, n=n)


def evaluate_transition_density(
    model: SupportsTransitionDensity,
    state_next: Any,
    state_previous: Any,
) -> Any:
    """Evaluate ``p(state_next | state_previous)`` on a transition model."""

    _require_capability(
        model,
        SupportsTransitionDensity,
        "transition_density(state_next, state_previous)",
    )
    return model.transition_density(state_next, state_previous)


def predict_distribution_from_model(
    model: SupportsPredictedDistribution,
    state_distribution: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Propagate ``state_distribution`` with a distribution-aware model."""

    _require_capability(
        model, SupportsPredictedDistribution, "predict_distribution(state_distribution)"
    )
    return model.predict_distribution(state_distribution, *args, **kwargs)


def require_model_attribute(model: object, *names: str) -> Any:
    """Return the first available attribute from ``names`` or raise."""

    if not names:
        raise ValueError("At least one attribute name is required.")

    for name in names:
        value = getattr(model, name, _MODEL_ATTRIBUTE_SENTINEL)
        if value is not _MODEL_ATTRIBUTE_SENTINEL:
            return value

    options = ", ".join(f"`{name}`" for name in names)
    raise AttributeError(
        f"{type(model).__name__} must expose one of the following attributes: {options}."
    )


def get_optional_model_attribute(
    model: object, *names: str, default: Any = None
) -> Any:
    """Return the first available optional model attribute from ``names``."""

    for name in names:
        value = getattr(model, name, _MODEL_ATTRIBUTE_SENTINEL)
        if value is not _MODEL_ATTRIBUTE_SENTINEL:
            return value

    return default


def linear_transition_arguments(model: object) -> LinearTransitionArguments:
    """Extract canonical arguments for a linear Gaussian prediction call."""

    return LinearTransitionArguments(
        system_matrix=require_model_attribute(model, "system_matrix"),
        sys_noise_cov=require_model_attribute(
            model, "system_noise_cov", "sys_noise_cov"
        ),
        sys_input=get_optional_model_attribute(model, "sys_input", "system_input"),
    )


def linear_measurement_arguments(model: object) -> LinearMeasurementArguments:
    """Extract canonical arguments for a linear Gaussian update call."""

    return LinearMeasurementArguments(
        measurement_matrix=require_model_attribute(model, "measurement_matrix"),
        meas_noise=require_model_attribute(
            model, "meas_noise", "measurement_noise_cov"
        ),
    )


__all__ = [
    "LinearMeasurementArguments",
    "LinearTransitionArguments",
    "as_density_transition_model",
    "as_likelihood_model",
    "as_sampleable_transition_model",
    "evaluate_likelihood",
    "evaluate_log_likelihood",
    "evaluate_transition_density",
    "get_optional_model_attribute",
    "linear_measurement_arguments",
    "linear_transition_arguments",
    "predict_distribution_from_model",
    "require_model_attribute",
    "sample_next_state",
]
