import unittest

import numpy as np
import numpy.testing as npt

import pyrecest.backend
from pyrecest.backend import array, eye, to_numpy
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import KalmanFilter
from pyrecest.filters._linear_gaussian import (
    huber_covariance_scale,
    linear_gaussian_update,
    linear_gaussian_update_robust,
    normalized_innovation_squared,
    student_t_covariance_scale,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
    reason="tests compare backend scalars with NumPy helpers",
)
class RobustLinearGaussianUpdateTest(unittest.TestCase):
    def setUp(self):
        self.mean = array([0.0, 0.0])
        self.covariance = eye(2)
        self.measurement_matrix = eye(2)
        self.meas_noise = eye(2)

    def test_normalized_innovation_squared_matches_mahalanobis_norm(self):
        innovation = array([2.0, 0.0])
        innovation_covariance = eye(2) * 2.0

        nis = normalized_innovation_squared(innovation, innovation_covariance)

        self.assertAlmostEqual(float(nis), 2.0)

    def test_student_t_scale_is_monotonic(self):
        small = student_t_covariance_scale(2.0, measurement_dim=2, dof=4.0)
        large = student_t_covariance_scale(100.0, measurement_dim=2, dof=4.0)

        self.assertEqual(float(small), 1.0)
        self.assertGreater(float(large), float(small))

    def test_huber_scale_uses_mahalanobis_threshold(self):
        inlier = huber_covariance_scale(4.0, huber_threshold=2.0)
        outlier = huber_covariance_scale(100.0, huber_threshold=2.0)

        self.assertEqual(float(inlier), 1.0)
        self.assertGreater(float(outlier), 1.0)

    def test_student_t_update_downweights_outlier(self):
        measurement = array([100.0, 100.0])
        plain_mean, _ = linear_gaussian_update(
            self.mean,
            self.covariance,
            measurement,
            self.measurement_matrix,
            self.meas_noise,
        )
        robust_mean, _, diagnostics = linear_gaussian_update_robust(
            self.mean,
            self.covariance,
            measurement,
            self.measurement_matrix,
            self.meas_noise,
            robust_method="student-t",
            student_t_dof=4.0,
            return_diagnostics=True,
        )

        self.assertGreater(float(diagnostics["scale"]), 1.0)
        self.assertLess(
            np.linalg.norm(to_numpy(robust_mean)),
            np.linalg.norm(to_numpy(plain_mean)),
        )

    def test_huber_update_downweights_outlier(self):
        measurement = array([100.0, 100.0])
        plain_mean, _ = linear_gaussian_update(
            self.mean,
            self.covariance,
            measurement,
            self.measurement_matrix,
            self.meas_noise,
        )
        robust_mean, _, diagnostics = linear_gaussian_update_robust(
            self.mean,
            self.covariance,
            measurement,
            self.measurement_matrix,
            self.meas_noise,
            robust_method="huber",
            huber_threshold=2.0,
            return_diagnostics=True,
        )

        self.assertGreater(float(diagnostics["scale"]), 1.0)
        self.assertLess(
            np.linalg.norm(to_numpy(robust_mean)),
            np.linalg.norm(to_numpy(plain_mean)),
        )

    def test_none_method_matches_standard_update(self):
        measurement = array([1.0, -1.0])
        plain_mean, plain_covariance = linear_gaussian_update(
            self.mean,
            self.covariance,
            measurement,
            self.measurement_matrix,
            self.meas_noise,
        )
        robust_mean, robust_covariance = linear_gaussian_update_robust(
            self.mean,
            self.covariance,
            measurement,
            self.measurement_matrix,
            self.meas_noise,
            robust_method="none",
        )

        npt.assert_allclose(to_numpy(robust_mean), to_numpy(plain_mean))
        npt.assert_allclose(to_numpy(robust_covariance), to_numpy(plain_covariance))

    def test_kalman_filter_update_linear_robust_returns_diagnostics(self):
        kf = KalmanFilter(GaussianDistribution(self.mean, self.covariance))
        diagnostics = kf.update_linear_robust(
            array([100.0, 100.0]),
            self.measurement_matrix,
            self.meas_noise,
            robust_method="student-t",
            return_diagnostics=True,
        )

        self.assertIn("nis", diagnostics)
        self.assertGreater(float(diagnostics["scale"]), 1.0)


if __name__ == "__main__":
    unittest.main()
