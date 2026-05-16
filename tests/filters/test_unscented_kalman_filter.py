import copy
import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, diag, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.unscented_kalman_filter import UnscentedKalmanFilter
from pyrecest.models import AdditiveNoiseMeasurementModel, AdditiveNoiseTransitionModel


class UnscentedKalmanFilterTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_initialization(self):
        filter_custom = UnscentedKalmanFilter(
            GaussianDistribution(array([1.0]), array([[10000.0]]))
        )
        npt.assert_allclose(filter_custom.get_point_estimate(), 1.0)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_initialization_gauss(self):
        filter_custom = UnscentedKalmanFilter(
            GaussianDistribution(array([4.0]), array([[10000.0]]))
        )
        npt.assert_allclose(filter_custom.get_point_estimate(), 4)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_update_linear_1d(self):
        kf = UnscentedKalmanFilter(GaussianDistribution(array([0.0]), array([[1.0]])))
        kf.update_identity(array([3.0]), array([[1.0]]))
        npt.assert_allclose(kf.get_point_estimate(), 1.5)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_update_linear_2d(self):
        filter_add = UnscentedKalmanFilter(
            GaussianDistribution(array([0.0, 1.0]), diag(array([1.0, 2.0])))
        )
        filter_id = copy.deepcopy(filter_add)
        gauss = GaussianDistribution(array([1.0, 0.0]), diag(array([2.0, 1.0])))
        filter_add.update_linear(gauss.mu, eye(2), gauss.C)
        filter_id.update_identity(gauss.mu, gauss.C)
        self.assertTrue(
            allclose(filter_add.get_point_estimate(), filter_id.get_point_estimate())
        )
        self.assertTrue(
            allclose(
                filter_add.filter_state.covariance(),
                filter_id.filter_state.covariance(),
            )
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_update_linear_rectangular_measurement(self):
        kf = UnscentedKalmanFilter(
            GaussianDistribution(array([0.0, 1.0]), diag(array([1.0, 2.0])))
        )
        kf.update_linear(array([1.0]), array([[1.0, 0.0]]), array([[0.5]]))

        npt.assert_allclose(kf.get_point_estimate(), array([2.0 / 3.0, 1.0]))
        npt.assert_allclose(
            kf.filter_state.covariance(), diag(array([1.0 / 3.0, 2.0]))
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_update_nonlinear_scalar_measurement(self):
        kf = UnscentedKalmanFilter(
            GaussianDistribution(array([0.0, 1.0]), diag(array([1.0, 2.0])))
        )

        def hx(x):
            return array([x[0] + x[1]])

        kf.update_nonlinear(array([2.0]), hx, array([[0.5]]))

        npt.assert_allclose(kf.get_point_estimate(), array([2.0 / 7.0, 11.0 / 7.0]))
        npt.assert_allclose(
            kf.filter_state.covariance(),
            array([[5.0 / 7.0, -4.0 / 7.0], [-4.0 / 7.0, 6.0 / 7.0]]),
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_predict_linear_2d(self):
        kf = UnscentedKalmanFilter(
            GaussianDistribution(array([0.0, 1.0]), diag(array([1.0, 2.0])))
        )
        kf.predict_linear(diag(array([1.0, 2.0])), diag(array([2.0, 1.0])))
        self.assertTrue(allclose(kf.get_point_estimate(), array([0.0, 2.0])))
        self.assertTrue(allclose(kf.filter_state.covariance(), diag(array([3.0, 9.0]))))
        kf.predict_linear(
            diag(array([1.0, 2.0])), diag(array([2.0, 1.0])), array([2.0, -2.0])
        )
        self.assertTrue(allclose(kf.get_point_estimate(), array([2.0, 2.0])))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_predict_model_matches_predict_nonlinear(self):
        initial_state = GaussianDistribution(
            array([1.0, -0.5]), diag(array([0.5, 0.25]))
        )
        direct = UnscentedKalmanFilter(initial_state)
        via_model = copy.deepcopy(direct)
        sys_noise_cov = diag(array([0.2, 0.1]))

        def fx(x, dt, bias=0.0):
            return array([x[0] + dt + bias, 2.0 * x[1]])

        direct.predict_nonlinear(fx, sys_noise_cov, dt=0.25, bias=0.1)
        transition_model = AdditiveNoiseTransitionModel(
            transition_function=fx,
            noise_distribution=GaussianDistribution(array([0.0, 0.0]), sys_noise_cov),
            dt=0.25,
            function_args={"bias": 0.1},
        )
        via_model.predict_model(transition_model)

        self.assertTrue(
            allclose(via_model.get_point_estimate(), direct.get_point_estimate())
        )
        self.assertTrue(
            allclose(
                via_model.filter_state.covariance(), direct.filter_state.covariance()
            )
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_update_model_matches_update_nonlinear(self):
        initial_state = GaussianDistribution(
            array([0.5, -1.0]), diag(array([0.75, 0.5]))
        )
        direct = UnscentedKalmanFilter(initial_state)
        via_model = copy.deepcopy(direct)
        meas_noise_cov = diag(array([0.3, 0.4]))
        measurement = array([1.2, -0.2])

        def hx(x, offset=0.0):
            return array([x[0] + offset, x[0] + x[1]])

        direct.update_nonlinear(measurement, hx, meas_noise_cov, offset=0.1)
        measurement_model = AdditiveNoiseMeasurementModel(
            measurement_function=hx,
            noise_distribution=GaussianDistribution(array([0.0, 0.0]), meas_noise_cov),
            function_args={"offset": 0.1},
        )
        via_model.update_model(measurement_model, measurement)

        self.assertTrue(
            allclose(via_model.get_point_estimate(), direct.get_point_estimate())
        )
        self.assertTrue(
            allclose(
                via_model.filter_state.covariance(), direct.filter_state.covariance()
            )
        )


if __name__ == "__main__":
    unittest.main()
