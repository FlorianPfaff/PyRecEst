import unittest

import numpy as np
import numpy.testing as npt

from pyrecest.backend import __backend_name__, array, eye, to_numpy
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import KalmanFilter
from pyrecest.filters._linear_gaussian import (
    linear_gaussian_predict,
    linear_gaussian_update,
)


@unittest.skipIf(
    __backend_name__ in ("pytorch", "jax"),
    reason="tests inspect NumPy eigenvalues and scalar conversions",
)
class LinearGaussianNumericsTest(unittest.TestCase):
    def test_update_preserves_symmetric_positive_semidefinite_covariance(self):
        mean = array([0.0, 0.0])
        covariance = array([[1.0, 0.999999], [0.999999, 1.0]])
        measurement = array([1.0])
        measurement_matrix = array([[1.0, -1.0]])
        meas_noise = array([[1e-9]])

        _, updated_covariance = linear_gaussian_update(
            mean,
            covariance,
            measurement,
            measurement_matrix,
            meas_noise,
        )
        updated_covariance_np = to_numpy(updated_covariance)

        npt.assert_allclose(updated_covariance_np, updated_covariance_np.T, atol=1e-10)
        self.assertGreaterEqual(np.linalg.eigvalsh(updated_covariance_np).min(), -1e-8)

    def test_predict_accepts_rectangular_state_transition(self):
        mean = array([1.0, -1.0])
        covariance = eye(2)
        system_matrix = array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
        sys_noise_cov = eye(3) * 0.1

        predicted_mean, predicted_covariance = linear_gaussian_predict(
            mean,
            covariance,
            system_matrix,
            sys_noise_cov,
        )

        self.assertEqual(predicted_mean.shape, (3,))
        self.assertEqual(predicted_covariance.shape, (3, 3))
        npt.assert_allclose(
            to_numpy(predicted_covariance),
            to_numpy(predicted_covariance).T,
            atol=1e-10,
        )

    def test_kalman_filter_validates_initial_state_shape(self):
        with self.assertRaises(ValueError):
            KalmanFilter((array([0.0, 1.0]), array([[1.0]])))

    def test_kalman_filter_accepts_scalar_one_dimensional_initial_state(self):
        kf = KalmanFilter((0.0, 1.0))

        self.assertEqual(kf.dim, 1)
        self.assertIsInstance(kf.filter_state, GaussianDistribution)


if __name__ == "__main__":
    unittest.main()
