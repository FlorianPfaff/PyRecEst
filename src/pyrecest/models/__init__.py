"""
The model package contains filter-independent objects that describe how states
evolve and how measurements are evaluated. These objects are deliberately small
and capability-oriented so filters can opt into the pieces they need.
"""

from .adapters import (
    LinearMeasurementArguments,
    LinearTransitionArguments,
    as_density_transition_model,
    as_likelihood_model,
    as_sampleable_transition_model,
    evaluate_likelihood,
    evaluate_log_likelihood,
    evaluate_transition_density,
    get_optional_model_attribute,
    linear_measurement_arguments,
    linear_transition_arguments,
    predict_distribution_from_model,
    require_model_attribute,
    sample_next_state,
)
from .additive_noise import AdditiveNoiseMeasurementModel, AdditiveNoiseTransitionModel
from .grid import (
    GridLikelihoodMeasurementModel,
    GridTransitionDensityFactoryModel,
    GridTransitionDensityModel,
)
from .likelihood import (
    DensityTransitionModel,
    LikelihoodMeasurementModel,
    SampleableTransitionModel,
    SupportsLikelihood,
    SupportsLogLikelihood,
    SupportsTransitionDensity,
    SupportsTransitionSampling,
)
from .linear_gaussian import (
    IdentityGaussianMeasurementModel,
    IdentityGaussianTransitionModel,
    LinearGaussianMeasurementModel,
    LinearGaussianTransitionModel,
)
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
    "GridLikelihoodMeasurementModel",
    "GridTransitionDensityFactoryModel",
    "GridTransitionDensityModel",
    "IdentityGaussianMeasurementModel",
    "IdentityGaussianTransitionModel",
    "LikelihoodMeasurementModel",
    "LinearGaussianMeasurementModel",
    "LinearGaussianTransitionModel",
    "LinearMeasurementArguments",
    "LinearTransitionArguments",
    "SampleableTransitionModel",
    "SupportsLikelihood",
    "SupportsLogLikelihood",
    "SupportsTransitionDensity",
    "SupportsTransitionSampling",
    "as_density_transition_model",
    "as_likelihood_model",
    "as_sampleable_transition_model",
    "evaluate_likelihood",
    "evaluate_log_likelihood",
    "evaluate_transition_density",
    "get_optional_model_attribute",
    "infer_state_dim_from_distribution",
    "linear_measurement_arguments",
    "linear_transition_arguments",
    "predict_distribution_from_model",
    "require_model_attribute",
    "sample_next_state",
    "validate_covariance_matrix",
    "validate_matrix",
    "validate_measurement_matrix",
    "validate_measurement_vector",
    "validate_noise_covariance",
    "validate_state_vector",
    "validate_transition_matrix",
    "validate_vector",
]
