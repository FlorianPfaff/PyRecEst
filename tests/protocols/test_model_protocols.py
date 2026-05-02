"""Tests for public model capability protocols."""

from __future__ import annotations

from pyrecest.backend import array
from pyrecest.models import (
    DensityTransitionModel,
    IdentityGaussianMeasurementModel,
    IdentityGaussianTransitionModel,
    LikelihoodMeasurementModel,
    SampleableTransitionModel,
    SupportsLikelihood as ModelsSupportsLikelihood,
    SupportsLinearGaussianMeasurement as ModelsSupportsLinearGaussianMeasurement,
    SupportsLinearGaussianTransition as ModelsSupportsLinearGaussianTransition,
    SupportsLogLikelihood as ModelsSupportsLogLikelihood,
    SupportsPredictedDistribution as ModelsSupportsPredictedDistribution,
    SupportsTransitionDensity as ModelsSupportsTransitionDensity,
    SupportsTransitionSampling as ModelsSupportsTransitionSampling,
)
from pyrecest.models.likelihood import (
    SupportsLikelihood as LikelihoodSupportsLikelihood,
)
from pyrecest.protocols.models import (
    SupportsLikelihood,
    SupportsLinearGaussianMeasurement,
    SupportsLinearGaussianTransition,
    SupportsLogLikelihood,
    SupportsPredictedDistribution,
    SupportsTransitionDensity,
    SupportsTransitionSampling,
)


def test_model_protocols_are_reexported_from_existing_model_locations():
    assert ModelsSupportsLikelihood is SupportsLikelihood
    assert LikelihoodSupportsLikelihood is SupportsLikelihood
    assert ModelsSupportsLogLikelihood is SupportsLogLikelihood
    assert ModelsSupportsTransitionSampling is SupportsTransitionSampling
    assert ModelsSupportsTransitionDensity is SupportsTransitionDensity
    assert ModelsSupportsPredictedDistribution is SupportsPredictedDistribution
    assert ModelsSupportsLinearGaussianTransition is SupportsLinearGaussianTransition
    assert ModelsSupportsLinearGaussianMeasurement is SupportsLinearGaussianMeasurement


def test_likelihood_model_satisfies_likelihood_protocols():
    model = LikelihoodMeasurementModel(
        lambda measurement, state: measurement + state,
        log_likelihood=lambda measurement, state: measurement - state,
    )

    assert isinstance(model, SupportsLikelihood)
    assert isinstance(model, SupportsLogLikelihood)


def test_transition_models_satisfy_transition_protocols():
    sampleable = SampleableTransitionModel(lambda state, n=1: state)
    density_based = DensityTransitionModel(
        lambda state_next, state_previous: state_next + state_previous
    )

    assert isinstance(sampleable, SupportsTransitionSampling)
    assert isinstance(density_based, SupportsTransitionDensity)


def test_linear_gaussian_models_satisfy_model_protocols():
    noise_covariance = array([[1.0, 0.0], [0.0, 1.0]])
    transition_model = IdentityGaussianTransitionModel(2, noise_covariance)
    measurement_model = IdentityGaussianMeasurementModel(2, noise_covariance)

    assert isinstance(transition_model, SupportsLinearGaussianTransition)
    assert isinstance(transition_model, SupportsPredictedDistribution)
    assert isinstance(measurement_model, SupportsLinearGaussianMeasurement)
    assert isinstance(measurement_model, SupportsPredictedDistribution)
