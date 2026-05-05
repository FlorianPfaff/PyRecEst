import unittest

import numpy.testing as npt

# pylint: disable=no-member,no-name-in-module
import pyrecest.backend
from pyrecest.backend import array, eye, isfinite, linalg, pi, sqrt
from pyrecest.filters import SphericalHarmonicsEOTTracker


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Spherical-harmonics EOT tracker currently uses pyshtools/numpy",
)
class TestSphericalHarmonicsEOTTracker(unittest.TestCase):
    def test_coefficient_vector_matrix_roundtrip(self):
        coefficients = array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        coeff_mat = SphericalHarmonicsEOTTracker.coefficients_to_matrix(coefficients)
        coefficients_roundtrip = SphericalHarmonicsEOTTracker.matrix_to_coefficients(coeff_mat)

        self.assertEqual(coeff_mat.shape, (3, 5))
        npt.assert_allclose(coefficients_roundtrip, coefficients)

    def test_real_complex_conversion_roundtrip_keeps_raw_coefficients(self):
        coefficients = array([0.5, 1.0, -2.0, 3.0, 4.0, -5.0, 6.0, -7.0, 8.0])
        coeff_mat = SphericalHarmonicsEOTTracker.coefficients_to_matrix(coefficients)

        complex_coeff_mat = SphericalHarmonicsEOTTracker._real_coeff_mat_to_complex(  # pylint: disable=protected-access
            coeff_mat
        )
        real_coeff_mat = SphericalHarmonicsEOTTracker._complex_coeff_mat_to_real(  # pylint: disable=protected-access
            complex_coeff_mat
        )

        npt.assert_allclose(real_coeff_mat, coeff_mat)

    def test_constant_radius_uses_unnormalized_spherical_harmonics_coefficients(self):
        tracker = SphericalHarmonicsEOTTracker(0, initial_radius=2.0)
        directions = array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ).T

        radii = tracker.evaluate_radius(directions)

        npt.assert_allclose(tracker.coefficients[0], 2.0 * sqrt(4.0 * pi))
        npt.assert_allclose(radii, array([2.0, 2.0, 2.0]), atol=1e-12)

    def test_measurement_function_projects_points_to_current_surface(self):
        tracker = SphericalHarmonicsEOTTracker(
            0,
            initial_radius=2.0,
            center=array([1.0, -2.0, 0.5]),
        )
        measurements = array(
            [
                [5.0, 1.0],
                [-2.0, -2.0],
                [0.5, -4.5],
            ]
        )

        predicted = tracker.measurement_function(tracker.get_point_estimate(), measurements)

        expected_points = array(
            [
                [3.0, 1.0],
                [-2.0, -2.0],
                [0.5, -1.5],
            ]
        )
        npt.assert_allclose(predicted, expected_points.T.reshape(-1), atol=1e-12)

    def test_update_increases_radius_toward_far_measurement(self):
        tracker = SphericalHarmonicsEOTTracker(
            0,
            initial_radius=1.0,
            coefficient_covariance=1.0,
            kinematic_covariance=1e-4,
        )
        prior_radius = tracker.evaluate_radius(array([1.0, 0.0, 0.0]))
        prior_covariance = tracker.covariance.copy()

        tracker.update(array([2.0, 0.0, 0.0]), meas_noise_cov=0.01 * eye(3))

        self.assertGreater(tracker.evaluate_radius(array([1.0, 0.0, 0.0])), prior_radius)
        self.assertLess(tracker.covariance[3, 3], prior_covariance[3, 3])
        self.assertTrue(isfinite(tracker.latest_predicted_measurement).all())
        self.assertTrue(linalg.eigvalsh(tracker.covariance)[0] > -1e-8)

    def test_update_accepts_measurements_by_row(self):
        tracker = SphericalHarmonicsEOTTracker(
            1,
            initial_radius=1.0,
            coefficient_covariance=0.5,
            kinematic_covariance=1e-4,
        )
        measurements_by_row = array(
            [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        )

        tracker.update(measurements_by_row, meas_noise_cov=0.01 * eye(6))

        self.assertEqual(tracker.get_point_estimate().shape, (7,))
        self.assertEqual(tracker.latest_predicted_measurement.shape, (6,))

    def test_zero_rotation_prediction_keeps_coefficients(self):
        tracker = SphericalHarmonicsEOTTracker(1, initial_radius=1.0)
        coefficients_prior = tracker.coefficients.copy()

        tracker.predict_rotation(0.0, 0.0, 0.0)

        npt.assert_allclose(tracker.coefficients, coefficients_prior)


if __name__ == "__main__":
    unittest.main()
