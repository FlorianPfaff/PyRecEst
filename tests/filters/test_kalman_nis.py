import unittest

from pyrecest.backend import allclose, array, diag, eye
from pyrecest.filters.kalman_filter import KalmanFilter


class KalmanNisTest(unittest.TestCase):
    def test_nis_and_state_is_unchanged(self):
        kf = KalmanFilter((array([1.0, -1.0]), diag(array([2.0, 3.0]))))
        mean_before = kf.get_point_estimate()
        cov_before = kf.filter_state.C

        innovation, innovation_covariance = kf.innovation_linear(
            array([4.0, 2.0]), eye(2), diag(array([5.0, 7.0]))
        )
        nis = kf.normalized_innovation_squared_linear(
            array([4.0, 2.0]), eye(2), diag(array([5.0, 7.0]))
        )

        self.assertTrue(allclose(innovation, array([3.0, 3.0])))
        self.assertTrue(allclose(innovation_covariance, diag(array([7.0, 10.0]))))
        self.assertTrue(allclose(nis, array(3.0**2 / 7.0 + 3.0**2 / 10.0)))
        self.assertTrue(allclose(kf.get_point_estimate(), mean_before))
        self.assertTrue(allclose(kf.filter_state.C, cov_before))


if __name__ == "__main__":
    unittest.main()
