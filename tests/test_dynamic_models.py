"""Tests for ready-made dynamic and sensor model catalog helpers."""

import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.backend import array, diag
from pyrecest.models import (
    bearing_only_measurement,
    camera_projection_measurement,
    constant_acceleration_transition_matrix,
    constant_velocity_model,
    constant_velocity_transition_matrix,
    continuous_to_discrete_lti,
    coordinated_turn_transition,
    fdoa_measurement,
    nearly_constant_speed_transition,
    radar_range_bearing_doppler_measurement,
    range_bearing_jacobian,
    range_bearing_measurement,
    range_bearing_model,
    se2_unicycle_transition,
    se3_pose_twist_transition,
    singer_model,
    tdoa_measurement,
    white_noise_acceleration_covariance,
)


class TestMotionModelCatalog(unittest.TestCase):
    def test_constant_velocity_matrix_and_covariance_use_derivative_grouped_order(self):
        transition = constant_velocity_transition_matrix(2.0, spatial_dim=2)
        covariance = white_noise_acceleration_covariance(
            2.0, spatial_dim=2, spectral_density=3.0
        )

        npt.assert_allclose(
            transition,
            np.array(
                [
                    [1.0, 0.0, 2.0, 0.0],
                    [0.0, 1.0, 0.0, 2.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )
        npt.assert_allclose(
            covariance,
            np.array(
                [
                    [8.0, 0.0, 6.0, 0.0],
                    [0.0, 8.0, 0.0, 6.0],
                    [6.0, 0.0, 6.0, 0.0],
                    [0.0, 6.0, 0.0, 6.0],
                ]
            ),
        )

    def test_linear_transition_models_predict_mean(self):
        model = constant_velocity_model(0.5, spatial_dim=2, spectral_density=0.1)
        npt.assert_allclose(
            model.predict_mean(array([1.0, 2.0, 4.0, -2.0])),
            np.array([3.0, 1.0, 4.0, -2.0]),
        )

        ca_transition = constant_acceleration_transition_matrix(2.0, spatial_dim=1)
        npt.assert_allclose(
            ca_transition, np.array([[1.0, 2.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])
        )

    def test_continuous_to_discrete_lti(self):
        transition, covariance = continuous_to_discrete_lti(
            np.array([[0.0, 1.0], [0.0, 0.0]]),
            np.array([[0.0], [1.0]]),
            np.array([[2.0]]),
            dt=1.0,
        )
        npt.assert_allclose(transition, np.array([[1.0, 1.0], [0.0, 1.0]]), atol=1e-12)
        npt.assert_allclose(
            covariance, np.array([[2.0 / 3.0, 1.0], [1.0, 2.0]]), atol=1e-12
        )

    def test_singer_model_shapes(self):
        model = singer_model(1.0, spatial_dim=2, tau=5.0, acceleration_variance=0.5)
        self.assertEqual(tuple(model.matrix.shape), (6, 6))
        self.assertEqual(tuple(model.noise_cov.shape), (6, 6))

    def test_nonlinear_motion_transitions(self):
        npt.assert_allclose(
            coordinated_turn_transition(array([0.0, 0.0, 1.0, 0.0, 0.0]), dt=2.0),
            np.array([2.0, 0.0, 1.0, 0.0, 0.0]),
        )
        npt.assert_allclose(
            nearly_constant_speed_transition(array([0.0, 0.0, 2.0, 0.0]), dt=3.0),
            np.array([6.0, 0.0, 2.0, 0.0]),
        )
        npt.assert_allclose(
            se2_unicycle_transition(array([0.0, 0.0, 0.0, 1.0, 0.0]), dt=2.0),
            np.array([2.0, 0.0, 0.0, 1.0, 0.0]),
        )

        state = array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 4.0, 5.0, 6.0, 0.01, 0.02, 0.03])
        npt.assert_allclose(
            se3_pose_twist_transition(state, dt=2.0)[:6],
            np.array([9.0, 12.0, 15.0, 0.12, 0.24, 0.36]),
        )


class TestSensorModelCatalog(unittest.TestCase):
    def test_range_bearing_and_jacobian(self):
        state = array([3.0, 4.0, 0.0, 0.0])
        npt.assert_allclose(
            range_bearing_measurement(state), np.array([5.0, np.arctan2(4.0, 3.0)])
        )
        npt.assert_allclose(
            bearing_only_measurement(state), np.array([np.arctan2(4.0, 3.0)])
        )
        npt.assert_allclose(
            range_bearing_jacobian(state),
            np.array([[0.6, 0.8, 0.0, 0.0], [-4.0 / 25.0, 3.0 / 25.0, 0.0, 0.0]]),
        )

        model = range_bearing_model(diag(array([0.1, 0.2])))
        npt.assert_allclose(
            model.evaluate(state), np.array([5.0, np.arctan2(4.0, 3.0)])
        )

    def test_radar_tdoa_fdoa_and_camera_measurements(self):
        radar_state = array([3.0, 4.0, 3.0, 4.0])
        npt.assert_allclose(
            radar_range_bearing_doppler_measurement(radar_state),
            np.array([5.0, np.arctan2(4.0, 3.0), 5.0]),
        )

        tdoa_state = array([0.0, 4.0, 0.0, 1.0])
        sensors = array([[0.0, 0.0], [3.0, 0.0]])
        npt.assert_allclose(tdoa_measurement(tdoa_state, sensors), np.array([1.0]))
        npt.assert_allclose(fdoa_measurement(tdoa_state, sensors), np.array([-0.2]))

        npt.assert_allclose(
            camera_projection_measurement(array([2.0, 4.0, 2.0])), np.array([1.0, 2.0])
        )
        camera_matrix = array([[2.0, 0.0, 10.0], [0.0, 3.0, 20.0], [0.0, 0.0, 1.0]])
        npt.assert_allclose(
            camera_projection_measurement(
                array([2.0, 4.0, 2.0]), camera_matrix=camera_matrix
            ),
            np.array([12.0, 26.0]),
        )


if __name__ == "__main__":
    unittest.main()
