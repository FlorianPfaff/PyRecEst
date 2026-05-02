"""Capability protocols for reusable transition and measurement models.

The model layer keeps system and measurement definitions independent from the
filters that consume them. The protocols in this module intentionally describe
small capabilities rather than forcing every model into one large inheritance
hierarchy.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TransitionModel(Protocol):
    """Protocol for deterministic state-transition models."""

    def transition_function(self, state: Any) -> Any:
        """Evaluate the noise-free state transition."""


@runtime_checkable
class MeasurementModel(Protocol):
    """Protocol for deterministic measurement models."""

    def measurement_function(self, state: Any) -> Any:
        """Evaluate the noise-free measurement function."""


@runtime_checkable
class DifferentiableTransitionModel(TransitionModel, Protocol):
    """Transition model that can provide a Jacobian at a state."""

    def jacobian(self, state: Any) -> Any:
        """Return the transition Jacobian evaluated at ``state``."""


@runtime_checkable
class DifferentiableMeasurementModel(MeasurementModel, Protocol):
    """Measurement model that can provide a Jacobian at a state."""

    def jacobian(self, state: Any) -> Any:
        """Return the measurement Jacobian evaluated at ``state``."""


@runtime_checkable
class SampleableTransitionModel(TransitionModel, Protocol):
    """Transition model that can draw successor-state samples."""

    def sample_next(self, state: Any, n: int = 1) -> Any:
        """Draw ``n`` samples from the transition distribution at ``state``."""


@runtime_checkable
class LikelihoodMeasurementModel(MeasurementModel, Protocol):
    """Measurement model that can evaluate a measurement likelihood."""

    def likelihood(self, measurement: Any, state: Any) -> Any:
        """Evaluate ``p(measurement | state)``."""
