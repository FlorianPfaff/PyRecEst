"""Base classes for reusable estimation model objects.

Model objects describe the probabilistic structure of an estimation problem.
They are intentionally independent from concrete filters: a transition model
encodes how a state evolves, a measurement model encodes how observations are
generated from states, and a noise model describes the uncertainty injected by
one of those mappings.

This module only provides lightweight marker classes and metadata. Concrete
model capabilities such as linear matrices, nonlinear functions, samplers,
densities, and likelihoods are expressed in :mod:`pyrecest.models.protocols`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .validation import validate_optional_dimension


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Optional descriptive metadata shared by reusable model objects.

    Parameters
    ----------
    name : str | None, optional
        Human-readable model name used in examples, diagnostics, or evaluation
        summaries. The name has no algorithmic meaning.
    state_dim : int | None, optional
        Dimension of state vectors accepted or produced by the model. A state
        vector follows the PyRecEst convention ``(state_dim,)`` for one state
        and ``(..., state_dim)`` for batched states where supported.
    measurement_dim : int | None, optional
        Dimension of measurement vectors produced or consumed by the model. A
        measurement vector follows the convention ``(measurement_dim,)`` for one
        measurement and ``(..., measurement_dim)`` for batched measurements
        where supported.
    """

    name: str | None = None
    state_dim: int | None = None
    measurement_dim: int | None = None

    def __post_init__(self) -> None:
        validate_optional_dimension("state_dim", self.state_dim)
        validate_optional_dimension("measurement_dim", self.measurement_dim)


class Model:
    """Marker base class for reusable, filter-independent model objects.

    Subclasses may implement any of the capability protocols in
    :mod:`pyrecest.models.protocols`. They should not depend on a concrete
    filter implementation. This keeps model objects usable by Kalman filters,
    sigma-point filters, particle filters, grid filters, simulations, and
    evaluation utilities whenever their capabilities match.
    """

    metadata: ModelMetadata = ModelMetadata()

    @property
    def name(self) -> str | None:
        """Human-readable model name, if available."""
        return self.metadata.name

    @property
    def state_dim(self) -> int | None:
        """Dimension of state vectors handled by the model, if known."""
        return self.metadata.state_dim

    @property
    def measurement_dim(self) -> int | None:
        """Dimension of measurement vectors handled by the model, if known."""
        return self.metadata.measurement_dim


class TransitionModel(Model):
    """Marker base class for models representing ``p(x_k | x_{k-1})``.

    Transition models describe how a previous state is mapped to a predicted
    state distribution. Depending on the concrete capabilities, they may expose
    linear transition matrices, nonlinear transition functions, transition
    densities, or state samplers.
    """


class MeasurementModel(Model):
    """Marker base class for models representing ``p(z_k | x_k)``.

    Measurement models describe how states generate measurements. Depending on
    the concrete capabilities, they may expose linear measurement matrices,
    nonlinear measurement functions, likelihood evaluators, or measurement
    samplers.
    """


class NoiseModel(Model):
    """Marker base class for process or measurement noise descriptions.

    A noise model may wrap an existing PyRecEst distribution, a covariance
    matrix, a sampler, or a density representation. Concrete behavior should be
    expressed through the capability protocols rather than through this marker
    class alone.
    """
