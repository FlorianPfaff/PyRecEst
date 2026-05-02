"""Capability protocols for reusable estimation model objects.

The protocols in this module are deliberately small. A concrete model can
implement only the capabilities it supports. Filters can then request exactly
what they need without coupling themselves to one concrete model hierarchy.

All array-like values are typed as :class:`typing.Any` to remain compatible with
PyRecEst's backend facade. Implementations should use ``pyrecest.backend``
arrays when backend portability is required.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

ArrayLike = Any
State = Any
Measurement = Any


@runtime_checkable
class HasStateDim(Protocol):
    """Object that declares a state dimension.

    The declared dimension refers to state vectors with shape ``(state_dim,)``
    and to batched states with trailing dimension ``state_dim`` where batching
    is supported.
    """

    @property
    def state_dim(self) -> int | None:
        """Dimension of state vectors handled by the object, if known."""
        ...


@runtime_checkable
class HasMeasurementDim(Protocol):
    """Object that declares a measurement dimension.

    The declared dimension refers to measurement vectors with shape
    ``(measurement_dim,)`` and to batched measurements with trailing dimension
    ``measurement_dim`` where batching is supported.
    """

    @property
    def measurement_dim(self) -> int | None:
        """Dimension of measurement vectors handled by the object, if known."""
        ...


@runtime_checkable
class HasLinearTransition(HasStateDim, Protocol):
    """Transition capability for linear Gaussian-style predictions.

    Attributes
    ----------
    system_matrix : ArrayLike
        State transition matrix with shape ``(state_dim, state_dim)``.
    system_noise_cov : ArrayLike
        Additive process-noise covariance with shape ``(state_dim, state_dim)``.
    sys_input : ArrayLike | None
        Optional additive system input with shape ``(state_dim,)``. Use ``None``
        when there is no deterministic input term.
    """

    system_matrix: ArrayLike
    system_noise_cov: ArrayLike
    sys_input: ArrayLike | None


@runtime_checkable
class HasLinearMeasurement(HasStateDim, HasMeasurementDim, Protocol):
    """Measurement capability for linear Gaussian-style updates.

    Attributes
    ----------
    measurement_matrix : ArrayLike
        Measurement matrix with shape ``(measurement_dim, state_dim)``.
    meas_noise : ArrayLike
        Measurement-noise covariance with shape
        ``(measurement_dim, measurement_dim)``.
    """

    measurement_matrix: ArrayLike
    meas_noise: ArrayLike


@runtime_checkable
class HasTransitionFunction(HasStateDim, Protocol):
    """Transition capability for deterministic nonlinear propagation.

    The transition function accepts a state vector with shape ``(state_dim,)``
    and returns a predicted state vector with the same convention. Implementers
    may additionally support batched states with trailing dimension
    ``state_dim``.
    """

    def transition_function(self, state: State) -> State:
        """Propagate ``state`` through the deterministic transition mapping."""
        ...


@runtime_checkable
class HasMeasurementFunction(HasStateDim, HasMeasurementDim, Protocol):
    """Measurement capability for deterministic nonlinear observations.

    The measurement function accepts a state vector with shape ``(state_dim,)``
    and returns a measurement vector with shape ``(measurement_dim,)``.
    Implementers may additionally support batched states.
    """

    def measurement_function(self, state: State) -> Measurement:
        """Map ``state`` to the corresponding deterministic measurement."""
        ...


@runtime_checkable
class HasTransitionJacobian(HasStateDim, Protocol):
    """Capability for transition Jacobians used by linearization-based filters."""

    def transition_jacobian(self, state: State) -> ArrayLike:
        """Return the Jacobian of the transition function at ``state``."""
        ...


@runtime_checkable
class HasMeasurementJacobian(HasStateDim, HasMeasurementDim, Protocol):
    """Capability for measurement Jacobians used by linearization-based filters."""

    def measurement_jacobian(self, state: State) -> ArrayLike:
        """Return the Jacobian of the measurement function at ``state``."""
        ...


@runtime_checkable
class HasTransitionSampler(HasStateDim, Protocol):
    """Transition capability for sampling predicted states.

    This is the natural interface for particle filters and simulation code. The
    sampler should return one next state for ``num_samples == 1`` or a batch of
    next states with trailing dimension ``state_dim`` for larger sample counts.
    """

    def sample_next(self, state: State, num_samples: int = 1) -> State:
        """Sample one or more next states conditioned on ``state``."""
        ...


@runtime_checkable
class HasTransitionDensity(HasStateDim, Protocol):
    """Transition capability for evaluating ``p(x_k | x_{k-1})``."""

    def transition_density(self, next_state: State, previous_state: State) -> ArrayLike:
        """Evaluate the transition density from ``previous_state`` to ``next_state``."""
        ...


@runtime_checkable
class HasLikelihood(HasStateDim, Protocol):
    """Measurement capability for evaluating ``p(z_k | x_k)``.

    The likelihood value may be a scalar or a backend array. Implementations may
    support vectorized evaluation over batches of states.
    """

    def likelihood(self, measurement: Measurement, state: State) -> ArrayLike:
        """Evaluate the measurement likelihood for ``measurement`` at ``state``."""
        ...
