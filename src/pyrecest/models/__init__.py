"""
The model package contains filter-independent objects that describe how states
evolve and how measurements are evaluated. These objects are deliberately small
and capability-oriented so filters can opt into the pieces they need.
"""

from pyrecest.protocols.models import (
    SupportsLikelihood,
    SupportsLinearGaussianMeasurement,
    SupportsLinearGaussianTransition,
    SupportsLogLikelihood,
    SupportsPredictedDistribution,
    SupportsTransitionDensity,
    SupportsTransitionSampling,
)

from .additive_noise import AdditiveNoiseMeasurementModel, AdditiveNoiseTransitionModel
from .likelihood import (
    DensityTransitionModel,
    LikelihoodMeasurementModel,
    SampleableTransitionModel,
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
    "IdentityGaussianMeasurementModel",
    "IdentityGaussianTransitionModel",
    "LikelihoodMeasurementModel",
    "LinearGaussianMeasurementModel",
    "LinearGaussianTransitionModel",
    "SampleableTransitionModel",
    "SupportsLikelihood",
    "SupportsLinearGaussianMeasurement",
    "SupportsLinearGaussianTransition",
    "SupportsLogLikelihood",
    "SupportsPredictedDistribution",
    "SupportsTransitionDensity",
    "SupportsTransitionSampling",
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
