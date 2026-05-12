import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member,duplicate-code
import pyrecest.backend
from pyrecest.backend import all, array, diag, eye, linalg, pi
from pyrecest.filters.vbrm_tracker import VBRMTracker, VbrmTracker


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="VBRM tracker tests currently use numpy.testing assertions",
)
class TestVBRMTracker(unittest.TestCase):
    def setUp(self):
        self.kinematic_state = array([0.0, 0.0, 1.0, -1.0])
        self.covariance = diag(array([0.1, 0.1, 0.01, 0.01]))
        self.shape_state = array([0.0, 2.0, 1.0])
        self.orientation_variance = 0.1
        self.measurement_matrix = array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        self.measurement_noise_cov = 0.01 * eye(2)
        self.tracker = VBRMTracker(
            self.kinematic_state,
            self.covariance,
            self.shape_state,
            self.orientation_variance,
            inverse_gamma_shape=10.0,
            measurement_noise_cov=self.measurement_noise_cov,
            measurement_matrix=self.measurement_matrix,
        )

    def test_initialization_and_alias(self):
        self.assertIs(VbrmTracker, VBRMTracker)
        npt.assert_allclose(self.tracker.kinematic_state, self.kinematic_state)
        npt.assert_allclose(self.tracker.covariance, self.covariance)
        npt.assert_allclose(self.tracker.get_point_estimate_shape(), self.shape_state)
        npt.assert_allclose(
            self.tracker.get_point_estimate_shape(full_axes=True),
            array([0.0, 4.0, 2.0]),
        )
        npt.assert_allclose(
            self.tracker.get_point_estimate_extent(),
            diag(array([4.0, 1.0])),
        )
        self.assertEqual(self.tracker.get_point_estimate().shape[0], 7)

    def test_extent_respects_orientation(self):
        tracker = VBRMTracker(
            self.kinematic_state,
            self.covariance,
            array([0.5 * pi, 2.0, 1.0]),
            self.orientation_variance,
            inverse_gamma_shape=10.0,
            measurement_noise_cov=self.measurement_noise_cov,
            measurement_matrix=self.measurement_matrix,
        )

        npt.assert_allclose(
            tracker.get_point_estimate_extent(),
            diag(array([1.0, 4.0])),
            atol=1e-12,
        )

    def test_predict_linear_moves_kinematics_and_forgets_shape(self):
        system_matrix = array(
            [
                [1.0, 0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0, 0.5],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        alpha_prior, beta_prior = self.tracker.get_inverse_gamma_parameters()

        self.tracker.predict_linear(
            system_matrix,
            0.01 * eye(4),
            orientation_sys_noise=0.01,
            forgetting_factor=0.95,
        )

        npt.assert_allclose(
            self.tracker.kinematic_state,
            array([0.5, -0.5, 1.0, -1.0]),
        )
        self.assertGreater(float(self.tracker.orientation_variance), self.orientation_variance)
        alpha_post, beta_post = self.tracker.get_inverse_gamma_parameters()
        npt.assert_allclose(alpha_post, 0.95 * alpha_prior)
        npt.assert_allclose(beta_post, 0.95 * beta_prior)

    def test_update_moves_centroid_and_updates_extent_parameters(self):
        tracker = VBRMTracker(
            array([0.0, 0.0, 0.0, 0.0]),
            diag(array([0.1, 0.1, 0.01, 0.01])),
            array([0.0, 1.0, 1.0]),
            0.1,
            inverse_gamma_shape=10.0,
            measurement_noise_cov=self.measurement_noise_cov,
            measurement_matrix=self.measurement_matrix,
        )
        alpha_prior, beta_prior = tracker.get_inverse_gamma_parameters()

        tracker.update(array([[2.0, 0.0], [2.1, 0.1], [1.9, -0.1]]))

        alpha_post, beta_post = tracker.get_inverse_gamma_parameters()
        self.assertGreater(tracker.kinematic_state[0], 0.0)
        self.assertGreater(float(alpha_post[0]), float(alpha_prior[0]))
        self.assertGreater(float(beta_post[0]), float(beta_prior[0]))
        self.assertTrue(all(linalg.eigvalsh(tracker.covariance) > 0.0))
        self.assertTrue(all(linalg.eigvalsh(tracker.get_point_estimate_extent()) > 0.0))

    def test_update_accepts_transposed_measurement_layout(self):
        tracker = VBRMTracker(
            array([0.0, 0.0, 0.0, 0.0]),
            diag(array([0.1, 0.1, 0.01, 0.01])),
            array([0.0, 1.0, 1.0]),
            0.1,
            inverse_gamma_shape=10.0,
            measurement_noise_cov=self.measurement_noise_cov,
            measurement_matrix=self.measurement_matrix,
        )

        tracker.update(array([[2.0, 0.0]]))

        self.assertGreater(tracker.kinematic_state[0], 0.0)

    def test_update_without_measurements_is_noop(self):
        prior_kinematic_state = self.tracker.kinematic_state.copy()
        prior_shape_state = self.tracker.get_point_estimate_shape().copy()

        self.tracker.update(array([[], []]))

        npt.assert_allclose(self.tracker.kinematic_state, prior_kinematic_state)
        npt.assert_allclose(self.tracker.get_point_estimate_shape(), prior_shape_state)

    def test_get_contour_points(self):
        contour_points = self.tracker.get_contour_points(4)

        self.assertEqual(contour_points.shape, (4, 2))
        npt.assert_allclose(contour_points[0], array([2.0, 0.0]))
        npt.assert_allclose(contour_points[1], array([0.0, 1.0]), atol=1e-12)

    def test_predict_constant_velocity(self):
        self.tracker.predict_constant_velocity(time_delta=0.5, sys_noise=0.01 * eye(4))

        npt.assert_allclose(
            self.tracker.kinematic_state,
            array([0.5, -0.5, 1.0, -1.0]),
        )


if __name__ == "__main__":
    unittest.main()
