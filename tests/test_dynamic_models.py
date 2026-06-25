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
    integrated_white_noise_covariance,
    kinematic_transition_matrix,
    nearly_constant_speed_transition,
    radar_range_bearing_doppler_measurement,
    range_bearing_jacobian,
    range_bearing_measurement,
    range_bearing_model,
    se2_unicycle_transition,
    se3_pose_twist_transition,
    singer_model,
    singer_process_noise_covariance,
    singer_transition_matrix,
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

    def test_kinematic_models_validate_integer_parameters(self):
        invalid_cases = [
            ({"spatial_dim": 0}, "spatial_dim"),
            ({"spatial_dim": 1.5}, "spatial_dim"),
            ({"spatial_dim": True}, "spatial_dim"),
            ({"spatial_dim": np.array([2])}, "spatial_dim"),
            ({"derivative_order": -1}, "derivative_order"),
            ({"derivative_order": 1.5}, "derivative_order"),
            ({"derivative_order": True}, "derivative_order"),
        ]

        for kwargs, message in invalid_cases:
            with self.subTest(case=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    kinematic_transition_matrix(1.0, **kwargs)

        transition = kinematic_transition_matrix(
            1.0,
            spatial_dim=np.array(1.0),
            derivative_order=np.array(1.0),
        )
        npt.assert_allclose(transition, np.array([[1.0, 1.0], [0.0, 1.0]]))

    def test_process_noise_rejects_invalid_parameters(self):
        invalid_cases = [
            {"dt": -1.0, "message": "dt"},
            {"dt": np.nan, "message": "dt"},
            {"dt": "1.0", "message": "dt"},
            {"spatial_dim": 1.5, "message": "spatial_dim"},
            {"spatial_dim": "2", "message": "spatial_dim"},
            {"derivative_order": 1.5, "message": "derivative_order"},
            {"derivative_order": b"1", "message": "derivative_order"},
            {"spectral_density": -1.0, "message": "spectral_density"},
            {"spectral_density": np.nan, "message": "spectral_density"},
            {"spectral_density": True, "message": "spectral_density"},
            {"spectral_density": "1.0", "message": "spectral_density"},
            {"spectral_density": ["1.0", "2.0"], "message": "spectral_density"},
            {"spectral_density": np.array([True, False], dtype=object), "message": "spectral_density"},
            {
                "spectral_density": np.array([1.0, np.nan]),
                "message": "spectral_density",
            },
        ]

        for case in invalid_cases:
            kwargs = {"dt": 1.0}
            kwargs.update(case)
            message = kwargs.pop("message")
            with self.subTest(case=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    integrated_white_noise_covariance(**kwargs)

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

    def test_continuous_to_discrete_lti_rejects_nonfinite_inputs(self):
        with self.assertRaisesRegex(ValueError, "dt"):
            continuous_to_discrete_lti(np.eye(2), dt=np.nan)

        with self.assertRaisesRegex(ValueError, "continuous_matrix"):
            continuous_to_discrete_lti(np.array([[0.0, np.nan], [0.0, 0.0]]))

        with self.assertRaisesRegex(ValueError, "noise_input_matrix"):
            continuous_to_discrete_lti(
                np.eye(2),
                np.array([[0.0], [np.nan]]),
                np.eye(1),
            )

        with self.assertRaisesRegex(ValueError, "continuous_noise_covariance"):
            continuous_to_discrete_lti(
                np.eye(2),
                np.array([[0.0], [1.0]]),
                np.array([[np.inf]]),
            )

    def test_singer_model_shapes(self):
        model = singer_model(1.0, spatial_dim=2, tau=5.0, acceleration_variance=0.5)
        self.assertEqual(tuple(model.matrix.shape), (6, 6))
        self.assertEqual(tuple(model.noise_cov.shape), (6, 6))

    def test_singer_models_validate_parameters(self):
        invalid_transition_cases = [
            {"dt": np.nan, "message": "dt"},
            {"spatial_dim": 1.5, "message": "spatial_dim"},
            {"tau": 0.0, "message": "tau"},
            {"tau": np.nan, "message": "tau"},
        ]
        for case in invalid_transition_cases:
            kwargs = {"dt": 1.0}
            kwargs.update(case)
            message = kwargs.pop("message")
            with self.subTest(case=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    singer_transition_matrix(**kwargs)

        invalid_noise_cases = [
            {"dt": -1.0, "message": "dt"},
            {"acceleration_variance": -1.0, "message": "acceleration_variance"},
            {"acceleration_variance": np.nan, "message": "acceleration_variance"},
            {
                "acceleration_variance": np.array([1.0, np.inf]),
                "message": "acceleration_variance",
            },
        ]
        for case in invalid_noise_cases:
            kwargs = {"dt": 1.0}
            kwargs.update(case)
            message = kwargs.pop("message")
            with self.subTest(case=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    singer_process_noise_covariance(**kwargs)

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

    def test_range_based_measurements_reject_zero_range(self):
        zero_range_state = array([0.0, 0.0, 1.0, 0.0])

        with self.assertRaises(ValueError):
            range_bearing_measurement(zero_range_state)

        with self.assertRaises(ValueError):
            range_bearing_jacobian(zero_range_state)

        with self.assertRaises(ValueError):
            radar_range_bearing_doppler_measurement(zero_range_state)

        with self.assertRaises(ValueError):
            fdoa_measurement(
                zero_range_state,
                array([[0.0, 0.0], [1.0, 0.0]]),
            )

    def test_sensor_models_validate_state_index_parameters(self):
        state = array([3.0, 4.0, 1.0, 2.0])
        invalid_position_indices = (
            (0.5, 1),
            (True, 1),
            ("0", 1),
            (b"0", 1),
            (-1, 1),
            (0, 4),
            (0, 1, 2),
            (0, 0),
        )
        for indices in invalid_position_indices:
            with self.subTest(indices=indices):
                with self.assertRaisesRegex(ValueError, "position_indices"):
                    range_bearing_measurement(state, position_indices=indices)

        with self.assertRaisesRegex(ValueError, "velocity_indices"):
            radar_range_bearing_doppler_measurement(
                state,
                velocity_indices=(2.5, 3),
            )

        with self.assertRaisesRegex(ValueError, "position_indices"):
            camera_projection_measurement(
                array([2.0, 4.0, 2.0]),
                position_indices=(0, 1, True),
            )

    def test_tdoa_fdoa_validate_reference_sensor_and_speed(self):
        state = array([0.0, 4.0, 0.0, 1.0])
        sensors = array([[0.0, 0.0], [3.0, 0.0]])

        for propagation_speed in (0.0, np.nan, np.inf, True, "1.0", b"1.0"):
            with self.subTest(propagation_speed=propagation_speed):
                with self.assertRaisesRegex(ValueError, "propagation_speed"):
                    tdoa_measurement(
                        state,
                        sensors,
                        propagation_speed=propagation_speed,
                    )
                with self.assertRaisesRegex(ValueError, "propagation_speed"):
                    fdoa_measurement(
                        state,
                        sensors,
                        propagation_speed=propagation_speed,
                    )

        for reference_sensor in (1.5, True, "0", b"0", -1, 2, np.array([0])):
            with self.subTest(reference_sensor=reference_sensor):
                with self.assertRaisesRegex(ValueError, "reference_sensor"):
                    tdoa_measurement(
                        state,
                        sensors,
                        reference_sensor=reference_sensor,
                    )
                with self.assertRaisesRegex(ValueError, "reference_sensor"):
                    fdoa_measurement(
                        state,
                        sensors,
                        reference_sensor=reference_sensor,
                    )

        invalid_sensor_positions = (
            array([0.0, 0.0]),
            array([[0.0, 0.0]]),
            array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )
        for sensor_positions in invalid_sensor_positions:
            with self.subTest(sensor_positions=sensor_positions):
                with self.assertRaisesRegex(ValueError, "sensor_positions"):
                    tdoa_measurement(state, sensor_positions)
                with self.assertRaisesRegex(ValueError, "sensor_positions"):
                    fdoa_measurement(state, sensor_positions)

        with self.assertRaisesRegex(ValueError, "sensor_velocities"):
            fdoa_measurement(
                state,
                sensors,
                sensor_velocities=array([[0.0, 0.0]]),
            )

    def test_camera_projection_rejects_zero_projection_scale(self):
        with self.assertRaisesRegex(ValueError, "camera depth"):
            camera_projection_measurement(array([2.0, 4.0, 0.0]))

        camera_matrix = array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "homogeneous camera scale"):
            camera_projection_measurement(
                array([2.0, 4.0, 2.0]),
                camera_matrix=camera_matrix,
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
