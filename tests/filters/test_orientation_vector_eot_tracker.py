import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member,duplicate-code
import pyrecest.backend
from pyrecest.backend import all, array, diag, eye, linalg, pi
from pyrecest.filters.orientation_vector_eot_tracker import (
    EOTOV0Tracker,
    EOTOVTracker,
    OrientationVectorEOT0Tracker,
    OrientationVectorEOTTracker,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="orientation-vector EOT tests currently use numpy.testing assertions",
)
class TestOrientationVectorEOTTracker(unittest.TestCase):
    def setUp(self):
        self.kinematic_state = array([0.0, 0.0, 1.0, 0.0])
        self.covariance = diag(array([0.1, 0.1, 0.01, 0.01]))
        self.shape_state = array([0.0, 2.0, 1.0])
        self.measurement_matrix = array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        self.measurement_noise_cov = 0.01 * eye(2)
        self.tracker = OrientationVectorEOTTracker(
            self.kinematic_state,
            self.covariance,
            self.shape_state,
            measurement_noise_cov=self.measurement_noise_cov,
            measurement_matrix=self.measurement_matrix,
            num_iterations=1,
        )

    def test_initialization_and_aliases(self):
        self.assertIs(EOTOVTracker, OrientationVectorEOTTracker)
        self.assertIs(OrientationVectorEOT0Tracker, EOTOV0Tracker)
        npt.assert_allclose(self.tracker.kinematic_state, self.kinematic_state)
        npt.assert_allclose(
            self.tracker.get_point_estimate_shape(),
            self.shape_state,
            atol=1e-12,
        )
        npt.assert_allclose(
            self.tracker.get_point_estimate_shape(full_axes=True),
            array([0.0, 4.0, 2.0]),
            atol=1e-12,
        )
        npt.assert_allclose(
            self.tracker.get_point_estimate_extent(),
            diag(array([4.0, 1.0])),
            atol=1e-12,
        )
        self.assertEqual(self.tracker.get_point_estimate().shape[0], 7)

    def test_extent_respects_orientation_vector(self):
        tracker = OrientationVectorEOTTracker(
            self.kinematic_state,
            self.covariance,
            array([0.5 * pi, 2.0, 1.0]),
            measurement_noise_cov=self.measurement_noise_cov,
            measurement_matrix=self.measurement_matrix,
        )

        npt.assert_allclose(
            tracker.get_point_estimate_extent(),
            diag(array([1.0, 4.0])),
            atol=1e-12,
        )

    def test_predict_constant_velocity_moves_state_and_forgets_shape(self):
        alpha_prior, beta_prior = self.tracker.get_inverse_gamma_parameters()

        self.tracker.predict_constant_velocity(
            time_delta=0.5,
            sys_noise=0.01 * eye(4),
            orientation_sys_noise=0.01,
            forgetting_factor=0.95,
        )

        npt.assert_allclose(
            self.tracker.kinematic_state,
            array([0.5, 0.0, 1.0, 0.0]),
        )
        alpha_post, beta_post = self.tracker.get_inverse_gamma_parameters()
        npt.assert_allclose(alpha_post, 0.95 * alpha_prior)
        npt.assert_allclose(beta_post, 0.95 * beta_prior)

    def test_update_moves_centroid_and_keeps_positive_extent(self):
        tracker = OrientationVectorEOTTracker(
            array([0.0, 0.0, 1.0, 0.0]),
            diag(array([0.1, 0.1, 0.01, 0.01])),
            array([0.0, 1.0, 1.0]),
            measurement_noise_cov=self.measurement_noise_cov,
            measurement_matrix=self.measurement_matrix,
            num_iterations=1,
        )
        alpha_prior, beta_prior = tracker.get_inverse_gamma_parameters()

        tracker.update(array([[2.0, 0.0], [2.1, 0.1], [1.9, -0.1]]))

        alpha_post, beta_post = tracker.get_inverse_gamma_parameters()
        self.assertGreater(tracker.kinematic_state[0], 0.0)
        self.assertGreater(float(alpha_post[0]), float(alpha_prior[0]))
        self.assertGreater(float(beta_post[0]), float(beta_prior[0]))
        self.assertTrue(all(linalg.eigvalsh(tracker.covariance) > 0.0))
        self.assertTrue(all(linalg.eigvalsh(tracker.get_point_estimate_extent()) > 0.0))

    def test_heading_constraint_changes_orientation_but_eotov0_does_not(self):
        measurements = array([[0.0, 2.0], [0.1, 2.1], [-0.1, 1.9]])
        constrained = OrientationVectorEOTTracker(
            array([0.0, 0.0, 1.0, 0.0]),
            diag(array([0.1, 0.1, 0.01, 0.01])),
            array([0.5 * pi, 1.0, 1.0]),
            measurement_noise_cov=self.measurement_noise_cov,
            measurement_matrix=self.measurement_matrix,
            heading_noise_variance=1e-6,
            num_iterations=1,
        )
        unconstrained = EOTOV0Tracker(
            array([0.0, 0.0, 1.0, 0.0]),
            diag(array([0.1, 0.1, 0.01, 0.01])),
            array([0.5 * pi, 1.0, 1.0]),
            measurement_noise_cov=self.measurement_noise_cov,
            measurement_matrix=self.measurement_matrix,
            num_iterations=1,
        )

        constrained.update(measurements)
        unconstrained.update(measurements)

        self.assertLess(abs(float(constrained.orientation_vector[1])), 0.8)
        self.assertGreater(abs(float(unconstrained.orientation_vector[1])), 0.8)

    def test_get_contour_points(self):
        contour_points = self.tracker.get_contour_points(4)

        self.assertEqual(contour_points.shape, (4, 2))
        npt.assert_allclose(contour_points[0], array([2.0, 0.0]))
        npt.assert_allclose(contour_points[1], array([0.0, 1.0]), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
