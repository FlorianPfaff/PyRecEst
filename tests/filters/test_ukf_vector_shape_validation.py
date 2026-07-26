import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, eye, reshape, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.unscented_kalman_filter import UnscentedKalmanFilter


@unittest.skipIf(
    pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
    reason="Not supported on this backend",
)
class UnscentedKalmanFilterVectorShapeValidationTest(unittest.TestCase):
    @staticmethod
    def _make_two_dimensional_filter():
        return UnscentedKalmanFilter(GaussianDistribution(array([0.0, 1.0]), eye(2)))

    def test_predict_rejects_matrix_transition_outputs(self):
        for output_shape in ((1, 2), (2, 1)):
            with self.subTest(output_shape=output_shape):
                kf = self._make_two_dimensional_filter()

                def fx(x, _dt):
                    return reshape(x, output_shape)

                with self.assertRaisesRegex(
                    ValueError,
                    "transition function output must be scalar or one-dimensional",
                ):
                    kf.predict_nonlinear(fx, zeros((2, 2)))

                npt.assert_allclose(kf.get_point_estimate(), array([0.0, 1.0]))
                npt.assert_allclose(kf.filter_state.covariance(), eye(2))

    def test_update_rejects_matrix_measurement_outputs(self):
        for output_shape in ((1, 2), (2, 1)):
            with self.subTest(output_shape=output_shape):
                kf = self._make_two_dimensional_filter()
                kf.predict_identity(zeros((2, 2)))

                def hx(x):
                    return reshape(x, output_shape)

                with self.assertRaisesRegex(
                    ValueError,
                    "measurement function output must be scalar or one-dimensional",
                ):
                    kf.update_nonlinear(array([0.0, 1.0]), hx, eye(2))

    def test_update_rejects_matrix_measurements(self):
        for measurement_shape in ((1, 2), (2, 1)):
            with self.subTest(measurement_shape=measurement_shape):
                kf = self._make_two_dimensional_filter()
                kf.predict_identity(zeros((2, 2)))
                measurement = reshape(array([0.0, 1.0]), measurement_shape)

                with self.assertRaisesRegex(
                    ValueError,
                    "measurement z must be scalar or one-dimensional",
                ):
                    kf.update_nonlinear(measurement, lambda x: x, eye(2))

    def test_scalar_transition_measurement_and_output_remain_supported(self):
        kf = UnscentedKalmanFilter(GaussianDistribution(array([0.0]), array([[1.0]])))

        kf.predict_nonlinear(lambda x, _dt: x[0] + 1.0, array([[0.0]]))
        kf.update_nonlinear(1.0, lambda x: x[0], 1.0)

        npt.assert_allclose(kf.get_point_estimate(), array([1.0]))
        npt.assert_allclose(kf.filter_state.covariance(), array([[0.5]]))


if __name__ == "__main__":
    unittest.main()
