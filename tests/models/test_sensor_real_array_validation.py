"""Regression tests for real-valued sensor-model array inputs."""

import unittest

import numpy as np
from pyrecest.models import (
    camera_projection_measurement,
    fdoa_measurement,
    radar_range_bearing_doppler_measurement,
    tdoa_measurement,
)


class TestSensorRealArrayValidation(unittest.TestCase):
    def test_tdoa_rejects_complex_state_vectors(self):
        state = np.asarray([1.0j, 4.0, 0.0, 1.0], dtype=np.complex128)
        sensors = np.asarray([[0.0, 0.0], [3.0, 0.0]])

        with self.assertRaisesRegex(ValueError, "state must contain real values"):
            tdoa_measurement(state, sensors)

    def test_tdoa_rejects_complex_sensor_positions(self):
        state = np.asarray([0.0, 4.0, 0.0, 1.0])
        sensors = np.asarray(
            [[0.0, 0.0], [3.0 + 1.0j, 0.0]],
            dtype=np.complex128,
        )

        with self.assertRaisesRegex(
            ValueError,
            "sensor_positions must contain real values",
        ):
            tdoa_measurement(state, sensors)

    def test_radar_rejects_complex_sensor_velocity(self):
        state = np.asarray([3.0, 4.0, 3.0, 4.0])
        sensor_velocity = np.asarray([1.0j, 0.0], dtype=np.complex128)

        with self.assertRaisesRegex(
            ValueError,
            "sensor_velocity must contain real values",
        ):
            radar_range_bearing_doppler_measurement(
                state,
                sensor_velocity=sensor_velocity,
            )

    def test_fdoa_rejects_complex_sensor_velocities(self):
        state = np.asarray([0.0, 4.0, 0.0, 1.0])
        sensors = np.asarray([[0.0, 0.0], [3.0, 0.0]])
        sensor_velocities = np.asarray(
            [[0.0, 0.0], [1.0j, 0.0]],
            dtype=np.complex128,
        )

        with self.assertRaisesRegex(
            ValueError,
            "sensor_velocities must contain real values",
        ):
            fdoa_measurement(
                state,
                sensors,
                sensor_velocities=sensor_velocities,
            )

    def test_camera_rejects_complex_matrix_dtype(self):
        state = np.asarray([2.0, 4.0, 2.0])
        camera_matrix = np.eye(3, dtype=np.complex128)

        with self.assertRaisesRegex(
            ValueError,
            "camera_matrix must contain real values",
        ):
            camera_projection_measurement(
                state,
                camera_matrix=camera_matrix,
            )


if __name__ == "__main__":
    unittest.main()
