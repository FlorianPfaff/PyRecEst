import numpy.testing as npt
from pyrecest.backend import eye, to_numpy
from pyrecest.models import (
    IdentityGaussianMeasurementModel,
    IdentityGaussianTransitionModel,
)


def test_identity_models_accept_scalar_noise_variance():
    transition_model = IdentityGaussianTransitionModel(2, 0.25)
    measurement_model = IdentityGaussianMeasurementModel(2, noise_covariance=0.5)

    npt.assert_allclose(
        to_numpy(transition_model.system_noise_cov),
        to_numpy(0.25 * eye(2)),
        atol=1e-12,
    )
    npt.assert_allclose(
        to_numpy(measurement_model.measurement_noise_cov),
        to_numpy(0.5 * eye(2)),
        atol=1e-12,
    )
