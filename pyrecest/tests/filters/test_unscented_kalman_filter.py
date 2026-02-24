import unittest
from pyrecest.filters.unscented_kalman_filter import UnscentedKalmanFilter
from pyrecest.distributions import GaussianDistribution
import copy
from pyrecest.backend import array, diag, eye, allclose
import numpy.testing as npt

class UnscentedKalmanFilterTest(unittest.TestCase):

    def test_initialization(self):
        filter_custom = UnscentedKalmanFilter(GaussianDistribution(array([1.0]), array([[10000.0]])))
        npt.assert_allclose(filter_custom.get_point_estimate(), 1.0)
    
    def test_initialization_gauss(self):
        filter_custom = UnscentedKalmanFilter(GaussianDistribution(array([4.0]), array([[10000.0]])))
        npt.assert_allclose(filter_custom.get_point_estimate(), 4)

    def test_update_linear_1d(self):
        kf = UnscentedKalmanFilter(GaussianDistribution(array([0.0]), array([[1.0]])))
        kf.update_identity(array([3.0]), array([[1.0]]))
        npt.assert_allclose(kf.get_point_estimate(), 1.5)

    def test_update_linear_2d(self):
        filter_add = UnscentedKalmanFilter(GaussianDistribution(array([0.0, 1.0]), diag(array([1.0, 2.0]))))
        filter_id = copy.deepcopy(filter_add)
        gauss = GaussianDistribution(array([1.0, 0.0]), diag(array([2.0, 1.0])))
        filter_add.update_linear(gauss.mu, eye(2), gauss.C)
        filter_id.update_identity(gauss.mu, gauss.C)
        self.assertTrue(allclose(filter_add.get_point_estimate(), filter_id.get_point_estimate()))
        self.assertTrue(allclose(filter_add.filter_state.covariance(), filter_id.filter_state.covariance()))

    def test_predict_linear_2d(self):
        kf = UnscentedKalmanFilter(GaussianDistribution(array([0.0, 1.0]), diag(array(array([1.0, 2.0])))))
        kf.predict_linear(diag(array([1.0, 2.0])), diag(array([2.0, 1.0])))
        self.assertTrue(allclose(kf.get_point_estimate(), array([0.0, 2.0])))
        self.assertTrue(allclose(kf.filter_state.covariance(), diag(array([3.0, 9.0]))))
        kf.predict_linear(diag(array([1.0, 2.0])), diag(array([2.0, 1.0])), array([2.0, -2.0]))
        self.assertTrue(allclose(kf.get_point_estimate(), array([2.0, 2.0])))

if __name__ == "__main__":
    unittest.main()