import unittest
import warnings

import numpy as np
from pyrecest.backend import array
from pyrecest.filters._linear_gaussian import (
    huber_covariance_scale,
    linear_gaussian_predict,
    linear_gaussian_update,
    normalized_innovation_squared,
    student_t_covariance_scale,
)


class FilterLinearGaussianComplexInputTest(unittest.TestCase):
    def assert_rejected_without_warnings(self, callback, message):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaisesRegex(ValueError, message):
                callback()
        self.assertEqual(caught, [])

    def test_prediction_rejects_complex_model_inputs(self):
        real_mean = array([1.0])
        real_covariance = array([[1.0]])
        real_matrix = array([[1.0]])
        zero_covariance = array([[0.0]])

        invalid_calls = (
            (
                "system_matrix must contain real-valued",
                lambda: linear_gaussian_predict(
                    real_mean,
                    real_covariance,
                    np.asarray([[1.0 + 2.0j]]),
                    zero_covariance,
                ),
            ),
            (
                "sys_noise_cov must contain real-valued",
                lambda: linear_gaussian_predict(
                    real_mean,
                    real_covariance,
                    real_matrix,
                    np.asarray([[1.0 + 2.0j]], dtype=object),
                ),
            ),
            (
                "sys_input must contain real-valued",
                lambda: linear_gaussian_predict(
                    real_mean,
                    real_covariance,
                    real_matrix,
                    zero_covariance,
                    np.asarray([1.0 + 2.0j]),
                ),
            ),
        )

        for message, callback in invalid_calls:
            with self.subTest(message=message):
                self.assert_rejected_without_warnings(callback, message)

    def test_update_rejects_complex_measurement_inputs(self):
        real_mean = array([0.0])
        real_covariance = array([[1.0]])
        real_matrix = array([[1.0]])
        real_noise = array([[1.0]])

        invalid_calls = (
            (
                "measurement must contain real-valued",
                lambda: linear_gaussian_update(
                    real_mean,
                    real_covariance,
                    np.asarray([1.0 + 2.0j]),
                    real_matrix,
                    real_noise,
                ),
            ),
            (
                "measurement_matrix must contain real-valued",
                lambda: linear_gaussian_update(
                    real_mean,
                    real_covariance,
                    array([1.0]),
                    np.asarray([[1.0 + 2.0j]], dtype=object),
                    real_noise,
                ),
            ),
            (
                "meas_noise must contain real-valued",
                lambda: linear_gaussian_update(
                    real_mean,
                    real_covariance,
                    array([1.0]),
                    real_matrix,
                    np.asarray([[1.0 + 2.0j]]),
                ),
            ),
        )

        for message, callback in invalid_calls:
            with self.subTest(message=message):
                self.assert_rejected_without_warnings(callback, message)

    def test_nis_paths_reject_complex_values(self):
        invalid_calls = (
            (
                "innovation must contain real-valued",
                lambda: normalized_innovation_squared(
                    np.asarray([1.0 + 2.0j]),
                    array([[1.0]]),
                ),
            ),
            (
                "finite and nonnegative",
                lambda: huber_covariance_scale(np.asarray(1.0 + 2.0j)),
            ),
            (
                "finite and nonnegative",
                lambda: student_t_covariance_scale(
                    np.asarray(1.0 + 2.0j),
                    measurement_dim=1,
                ),
            ),
        )

        for message, callback in invalid_calls:
            with self.subTest(message=message):
                self.assert_rejected_without_warnings(callback, message)


if __name__ == "__main__":
    unittest.main()
