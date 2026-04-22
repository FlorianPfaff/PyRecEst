import copy
import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, diag, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.kalman_filter import KalmanFilter


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


if __name__ == "__main__":
    unittest.main()
