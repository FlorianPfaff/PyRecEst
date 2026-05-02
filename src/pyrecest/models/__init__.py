"""Reusable model object interfaces for PyRecEst.

The :mod:`pyrecest.models` package contains filter-independent contracts for
transition, measurement, and noise models. The initial model layer is additive:
it does not change existing filter APIs, but it gives future filters and
examples a common vocabulary for model capabilities.
"""

from .base import MeasurementModel, Model, ModelMetadata, NoiseModel, TransitionModel
from .protocols import (
    ArrayLike,
    HasLikelihood,
    HasLinearMeasurement,
    HasLinearTransition,
    HasMeasurementDim,
    HasMeasurementFunction,
    HasMeasurementJacobian,
    HasStateDim,
    HasTransitionDensity,
    HasTransitionFunction,
    HasTransitionJacobian,
    HasTransitionSampler,
    Measurement,
    State,
)
from .validation import require_capability, validate_optional_dimension

__all__ = [
    "ArrayLike",
    "HasLikelihood",
    "HasLinearMeasurement",
    "HasLinearTransition",
    "HasMeasurementDim",
    "HasMeasurementFunction",
    "HasMeasurementJacobian",
    "HasStateDim",
    "HasTransitionDensity",
    "HasTransitionFunction",
    "HasTransitionJacobian",
    "HasTransitionSampler",
    "Measurement",
    "MeasurementModel",
    "Model",
    "ModelMetadata",
    "NoiseModel",
    "State",
    "TransitionModel",
    "require_capability",
    "validate_optional_dimension",
]
