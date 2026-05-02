"""
The model package contains filter-independent objects that describe how states
evolve and how measurements are evaluated. These objects are deliberately small
and capability-oriented so filters can opt into the pieces they need.
"""

from .additive_noise import AdditiveNoiseMeasurementModel, AdditiveNoiseTransitionModel
from .likelihood import (
    DensityTransitionModel,
    LikelihoodMeasurementModel,
    SampleableTransitionModel,
    SupportsLikelihood,
    SupportsLogLikelihood,
    SupportsTransitionDensity,
    SupportsTransitionSampling,
)
from .linear_gaussian import IdentityGaussianMeasurementModel
from .linear_gaussian import IdentityGaussianTransitionModel
from .linear_gaussian import LinearGaussianMeasurementModel
from .linear_gaussian import LinearGaussianTransitionModel
from .validation import (
    infer_state_dim_from_distribution,
    validate_covariance_matrix,
    validate_matrix,
    validate_measurement_matrix,
    validate_measurement_vector,
    validate_noise_covariance,
    validate_state_vector,
    validate_transition_matrix,
    validate_vector,
)

__all__ = [
    "AdditiveNoiseMeasurementModel",
    "AdditiveNoiseTransitionModel",
    "DensityTransitionModel",
    "LikelihoodMeasurementModel",
    "SampleableTransitionModel",
    "SupportsLikelihood",
    "SupportsLogLikelihood",
    "SupportsTransitionDensity",
    "SupportsTransitionSampling",
    "IdentityGaussianMeasurementModel",
    "IdentityGaussianTransitionModel",
    "LinearGaussianMeasurementModel",
    "LinearGaussianTransitionModel",
    "infer_state_dim_from_distribution",
    "validate_covariance_matrix",
    "validate_matrix",
    "validate_measurement_matrix",
    "validate_measurement_vector",
    "validate_noise_covariance",
    "validate_state_vector",
    "validate_transition_matrix",
    "validate_vector",
]
