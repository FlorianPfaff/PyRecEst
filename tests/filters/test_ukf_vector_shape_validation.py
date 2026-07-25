import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.unscented_kalman_filter import UnscentedKalmanFilter


@unittest.skipIf(
    pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
    reason="Not supported on this backend",
)
class UnscentedKalmanFilterVectorShapeTest(unittest.TestCase):
    @staticmethod
    def _two_dimensional_filter():
        return UnscentedKalmanFilter(
            GaussianDistribution(array([0.0, 0.0]), eye(2))
        )

    def test_predict_rejects_matrix_transition_outputs(self):
        matrix_outputs = (
            array([[0.0, 0.0]]),
            array([[0.0], [0.0]]),
        )

        for output in matrix_outputs:
            with self.subTest(shape=output.shape):
                kf = self._two_dimensional_filter()

                def fx(_x, _dt, output=output):
                    return output

                with self.assertRaisesRegex(
                    ValueError,
                    "transition function output must be a scalar or one-dimensional vector",
                ):
                    kf.predict_nonlinear(fx, eye(2), dt=1.0)

    def test_update_rejects_matrix_measurement_function_outputs(self):
        matrix_outputs = (
            array([[0.0, 0.0]]),
            array([[0.0], [0.0]]),
        )

        for output in matrix_outputs:
            with self.subTest(shape=output.shape):
                kf = self._two_dimensional_filter()

                def hx(_x, output=output):
                    return output

                with self.assertRaisesRegex(
                    ValueError,
                    "measurement function output must be a scalar or one-dimensional vector",
                ):
                    kf.update_nonlinear(array([0.0, 0.0]), hx, eye(2))

    def test_update_rejects_matrix_measurements(self):
        matrix_measurements = (
            array([[0.0, 0.0]]),
            array([[0.0], [0.0]]),
        )

        for measurement in matrix_measurements:
            with self.subTest(shape=measurement.shape):
                kf = self._two_dimensional_filter()

                def hx(x):
                    return x

                with self.assertRaisesRegex(
                    ValueError,
                    "measurement z must be a scalar or one-dimensional vector",
                ):
                    kf.update_nonlinear(measurement, hx, eye(2))

    def test_scalar_outputs_and_measurements_remain_supported(self):
        kf = UnscentedKalmanFilter(
            GaussianDistribution(array([0.0]), array([[1.0]]))
        )

        def fx(x, _dt):
            return x[0]

        def hx(x):
            return x[0]

        kf.predict_nonlinear(fx, array([[0.0]]), dt=1.0)
        kf.update_nonlinear(1.0, hx, array([[1.0]]))

        npt.assert_allclose(kf.get_point_estimate(), array([0.5]), atol=1e-10)
        self.assertEqual(kf.get_point_estimate().shape, (1,))


if __name__ == "__main__":
    unittest.main()
