import copy
import unittest

import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.unscented_kalman_filter import UnscentedKalmanFilter
from pyrecest.models import AdditiveNoiseMeasurementModel, AdditiveNoiseTransitionModel


@unittest.skipIf(
    pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
    reason="UnscentedKalmanFilter is not supported on this backend.",
)
class UnscentedKalmanFilterNoiseMeanTest(unittest.TestCase):
    def test_predict_model_applies_process_noise_mean(self):
        initial_state = GaussianDistribution(array([1.0]), array([[1.5]]))
        direct = UnscentedKalmanFilter(initial_state)
        via_model = copy.deepcopy(direct)
        noise_mean = array([2.0])
        noise_covariance = array([[0.5]])

        direct.predict_nonlinear(
            lambda state, _dt: state + noise_mean,
            noise_covariance,
        )
        via_model.predict_model(
            AdditiveNoiseTransitionModel(
                lambda state: state,
                noise_mean=noise_mean,
                noise_covariance=noise_covariance,
            )
        )

        npt.assert_allclose(via_model.get_point_estimate(), array([3.0]), atol=1e-8)
        npt.assert_allclose(
            via_model.get_point_estimate(),
            direct.get_point_estimate(),
            rtol=1e-10,
            atol=1e-10,
        )
        npt.assert_allclose(
            via_model.filter_state.covariance(),
            direct.filter_state.covariance(),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_update_model_applies_measurement_noise_mean(self):
        initial_state = GaussianDistribution(array([0.0]), array([[1.0]]))
        direct = UnscentedKalmanFilter(initial_state)
        via_model = copy.deepcopy(direct)
        noise_mean = array([2.0])
        noise_covariance = array([[1.0]])
        measurement = array([3.0])

        direct.update_nonlinear(
            measurement,
            lambda state: state + noise_mean,
            noise_covariance,
        )
        via_model.update_model(
            AdditiveNoiseMeasurementModel(
                lambda state: state,
                noise_mean=noise_mean,
                noise_covariance=noise_covariance,
            ),
            measurement,
        )

        npt.assert_allclose(via_model.get_point_estimate(), array([0.5]), atol=1e-8)
        npt.assert_allclose(
            via_model.get_point_estimate(),
            direct.get_point_estimate(),
            rtol=1e-10,
            atol=1e-10,
        )
        npt.assert_allclose(
            via_model.filter_state.covariance(),
            direct.filter_state.covariance(),
            rtol=1e-10,
            atol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
