import copy
import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, diag, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.kalman_filter import KalmanFilter


class _LinearGaussianTransitionModel:
    def __init__(self, system_matrix, system_noise_cov, sys_input=None):
        self.system_matrix = system_matrix
        self.system_noise_cov = system_noise_cov
        if sys_input is not None:
            self.sys_input = sys_input


class _LinearGaussianTransitionModelWithAliases:
    def __init__(self, system_matrix, sys_noise_cov, system_input=None):
        self.system_matrix = system_matrix
        self.sys_noise_cov = sys_noise_cov
        if system_input is not None:
            self.system_input = system_input


class _LinearGaussianMeasurementModel:
    def __init__(self, measurement_matrix, meas_noise):
        self.measurement_matrix = measurement_matrix
        self.meas_noise = meas_noise


class KalmanFilterTest(unittest.TestCase):
    def test_initialization_mean_cov(self):
        filter_custom = KalmanFilter((array([1]), array([[10000]])))
        self.assertTrue(allclose(filter_custom.get_point_estimate(), array([1])))

    def test_initialization_gauss(self):
        filter_custom = KalmanFilter(
            initial_state=GaussianDistribution(array([4]), array([[10000]]))
        )
        self.assertTrue(allclose(filter_custom.get_point_estimate(), array([4])))

    def test_update_with_likelihood_1d(self):
        kf = KalmanFilter((array([0]), array([[1]])))
        kf.update_identity(array(1), array(3))
        self.assertTrue(allclose(kf.get_point_estimate(), array([1.5])))

    def test_update_with_meas_noise_and_meas_1d(self):
        kf = KalmanFilter((array([0]), array([[1]])))
        kf.update_identity(array(1), array(4))
        self.assertTrue(allclose(kf.filter_state.C, array([[0.5]])))
        self.assertTrue(allclose(kf.get_point_estimate(), array([2])))

    def test_update_linear_2d(self):
        filter_add = KalmanFilter((array([0.0, 1.0]), diag(array([1.0, 2.0]))))
        filter_id = copy.deepcopy(filter_add)
        gauss = GaussianDistribution(array([1.0, 0.0]), diag(array([2.0, 1.0])))
        filter_add.update_linear(gauss.mu, eye(2), gauss.C)
        filter_id.update_identity(gauss.C, gauss.mu)
        self.assertTrue(
            allclose(filter_add.get_point_estimate(), filter_id.get_point_estimate())
        )
        self.assertTrue(allclose(filter_add.filter_state.C, filter_id.filter_state.C))

    def test_update_model_matches_update_linear_2d(self):
        initial_state = (array([0.0, 1.0]), diag(array([1.0, 2.0])))
        filter_linear = KalmanFilter(initial_state)
        filter_model = KalmanFilter(initial_state)

        measurement = array([1.0, 0.0])
        measurement_matrix = eye(2)
        meas_noise = diag(array([2.0, 1.0]))
        measurement_model = _LinearGaussianMeasurementModel(
            measurement_matrix,
            meas_noise,
        )

        filter_linear.update_linear(measurement, measurement_matrix, meas_noise)
        filter_model.update_model(measurement_model, measurement)

        self.assertTrue(
            allclose(
                filter_linear.get_point_estimate(), filter_model.get_point_estimate()
            )
        )
        self.assertTrue(
            allclose(filter_linear.filter_state.C, filter_model.filter_state.C)
        )

    def test_update_linear_default_return_stays_none(self):
        kf = KalmanFilter((array([0.0]), array([[1.0]])))
        result = kf.update_linear(array([2.0]), array([[1.0]]), array([[1.0]]))
        self.assertIsNone(result)

    def test_update_linear_can_return_diagnostics(self):
        kf = KalmanFilter((array([0.0, 1.0]), diag(array([1.0, 2.0]))))
        diagnostics = kf.update_linear(
            array([2.0, 0.0]),
            eye(2),
            diag(array([2.0, 1.0])),
            return_diagnostics=True,
        )

        self.assertEqual(diagnostics["action"], "updated")
        self.assertEqual(diagnostics["scale"], 1.0)
        self.assertTrue(allclose(diagnostics["residual"], array([2.0, -1.0])))
        self.assertTrue(allclose(diagnostics["nis"], array(5.0 / 3.0)))

    def test_update_linear_scale_is_applied_and_reported(self):
        kf = KalmanFilter((array([0.0]), array([[1.0]])))
        diagnostics = kf.update_linear(
            array([4.0]),
            array([[1.0]]),
            array([[1.0]]),
            return_diagnostics=True,
            scale=4.0,
            action="huberized",
        )

        self.assertEqual(diagnostics["action"], "huberized")
        self.assertEqual(diagnostics["scale"], 4.0)
        self.assertTrue(allclose(diagnostics["nis"], array(8.0)))
        self.assertTrue(allclose(kf.get_point_estimate(), array([0.8])))
        self.assertTrue(allclose(kf.filter_state.C, array([[0.8]])))

    def test_update_identity_forwards_diagnostics(self):
        kf = KalmanFilter((array([0.0]), array([[1.0]])))
        diagnostics = kf.update_identity(
            array([[1.0]]),
            array([2.0]),
            return_diagnostics=True,
            action="accepted",
        )

        self.assertEqual(diagnostics["action"], "accepted")
        self.assertTrue(allclose(diagnostics["residual"], array([2.0])))

    def test_update_model_forwards_diagnostics(self):
        kf = KalmanFilter((array([0.0]), array([[1.0]])))
        measurement_model = _LinearGaussianMeasurementModel(
            array([[1.0]]),
            array([[1.0]]),
        )
        diagnostics = kf.update_model(
            measurement_model,
            array([2.0]),
            return_diagnostics=True,
            action="accepted",
        )

        self.assertEqual(diagnostics["action"], "accepted")
        self.assertTrue(allclose(diagnostics["residual"], array([2.0])))

    def test_update_linear_robust_none_matches_update_linear_2d(self):
        initial_state = (array([0.0, 1.0]), diag(array([1.0, 2.0])))
        filter_linear = KalmanFilter(initial_state)
        filter_robust = KalmanFilter(initial_state)

        measurement = array([1.0, 0.0])
        measurement_matrix = eye(2)
        meas_noise = diag(array([2.0, 1.0]))

        filter_linear.update_linear(measurement, measurement_matrix, meas_noise)
        diagnostics = filter_robust.update_linear_robust(
            measurement,
            measurement_matrix,
            meas_noise,
            robust_update=None,
            return_diagnostics=True,
        )

        self.assertTrue(diagnostics["accepted"])
        self.assertEqual(diagnostics["action"], "updated")
        self.assertEqual(diagnostics["scale"], 1.0)
        self.assertTrue(
            allclose(
                filter_linear.get_point_estimate(), filter_robust.get_point_estimate()
            )
        )
        self.assertTrue(
            allclose(filter_linear.filter_state.C, filter_robust.filter_state.C)
        )

    def test_update_linear_robust_rejects_when_gated(self):
        kf = KalmanFilter((array([0.0]), array([[1.0]])))

        diagnostics = kf.update_linear_robust(
            array([10.0]),
            array([[1.0]]),
            array([[1.0]]),
            robust_update=None,
            gate_threshold=1.0,
            return_diagnostics=True,
        )

        self.assertFalse(diagnostics["accepted"])
        self.assertEqual(diagnostics["action"], "rejected")
        self.assertTrue(allclose(kf.get_point_estimate(), array([0.0])))
        self.assertTrue(allclose(kf.filter_state.C, array([[1.0]])))

    def test_student_t_robust_update_downweights_outlier(self):
        filter_linear = KalmanFilter((array([0.0]), array([[1.0]])))
        filter_robust = KalmanFilter((array([0.0]), array([[1.0]])))

        measurement = array([100.0])
        measurement_matrix = array([[1.0]])
        meas_noise = array([[1.0]])

        filter_linear.update_linear(measurement, measurement_matrix, meas_noise)
        diagnostics = filter_robust.update_linear_robust(
            measurement,
            measurement_matrix,
            meas_noise,
            robust_update="student-t",
            student_t_dof=4.0,
            return_diagnostics=True,
        )

        self.assertTrue(diagnostics["accepted"])
        self.assertEqual(diagnostics["action"], "student_t")
        self.assertGreater(diagnostics["scale"], 1.0)
        self.assertLess(
            float(filter_robust.get_point_estimate()[0]),
            float(filter_linear.get_point_estimate()[0]),
        )

    def test_huber_robust_update_downweights_outlier(self):
        filter_linear = KalmanFilter((array([0.0]), array([[1.0]])))
        filter_robust = KalmanFilter((array([0.0]), array([[1.0]])))

        measurement = array([100.0])
        measurement_matrix = array([[1.0]])
        meas_noise = array([[1.0]])

        filter_linear.update_linear(measurement, measurement_matrix, meas_noise)
        diagnostics = filter_robust.update_linear_robust(
            measurement,
            measurement_matrix,
            meas_noise,
            robust_update="huber",
            huber_threshold=2.0,
            return_diagnostics=True,
        )

        self.assertTrue(diagnostics["accepted"])
        self.assertEqual(diagnostics["action"], "huberized")
        self.assertGreater(diagnostics["scale"], 1.0)
        self.assertLess(
            float(filter_robust.get_point_estimate()[0]),
            float(filter_linear.get_point_estimate()[0]),
        )

    def test_predict_identity_1d(self):
        kf = KalmanFilter((array([0]), array([[1]])))
        kf.predict_identity(array([[3]]), array([1]))
        self.assertTrue(allclose(kf.get_point_estimate(), array([1])))
        self.assertTrue(allclose(kf.filter_state.C, array([[4]])))

    def test_predict_linear_2d(self):
        kf = KalmanFilter((array([0, 1]), diag(array([1, 2]))))
        kf.predict_linear(diag(array([1, 2])), diag(array([2, 1])))
        self.assertTrue(allclose(kf.get_point_estimate(), array([0, 2])))
        self.assertTrue(allclose(kf.filter_state.C, diag(array([3, 9]))))

        kf.predict_linear(diag(array([1, 2])), diag(array([2, 1])), array([2, -2]))
        self.assertTrue(allclose(kf.get_point_estimate(), array([2, 2])))

    def test_predict_model_matches_predict_linear_2d(self):
        initial_state = (array([0.0, 1.0]), diag(array([1.0, 2.0])))
        filter_linear = KalmanFilter(initial_state)
        filter_model = KalmanFilter(initial_state)

        system_matrix = diag(array([1.0, 2.0]))
        sys_noise_cov = diag(array([2.0, 1.0]))
        sys_input = array([2.0, -2.0])
        transition_model = _LinearGaussianTransitionModel(
            system_matrix,
            sys_noise_cov,
            sys_input,
        )

        filter_linear.predict_linear(system_matrix, sys_noise_cov, sys_input)
        filter_model.predict_model(transition_model)

        self.assertTrue(
            allclose(
                filter_linear.get_point_estimate(), filter_model.get_point_estimate()
            )
        )
        self.assertTrue(
            allclose(filter_linear.filter_state.C, filter_model.filter_state.C)
        )

    def test_predict_model_accepts_existing_noise_and_input_aliases(self):
        initial_state = (array([0.0, 1.0]), diag(array([1.0, 2.0])))
        filter_linear = KalmanFilter(initial_state)
        filter_model = KalmanFilter(initial_state)

        system_matrix = diag(array([1.0, 2.0]))
        sys_noise_cov = diag(array([2.0, 1.0]))
        system_input = array([2.0, -2.0])
        transition_model = _LinearGaussianTransitionModelWithAliases(
            system_matrix,
            sys_noise_cov,
            system_input,
        )

        filter_linear.predict_linear(system_matrix, sys_noise_cov, system_input)
        filter_model.predict_model(transition_model)

        self.assertTrue(
            allclose(
                filter_linear.get_point_estimate(), filter_model.get_point_estimate()
            )
        )
        self.assertTrue(
            allclose(filter_linear.filter_state.C, filter_model.filter_state.C)
        )


if __name__ == "__main__":
    unittest.main()
