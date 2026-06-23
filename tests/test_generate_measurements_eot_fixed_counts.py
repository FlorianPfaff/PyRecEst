import unittest

import numpy as np
from pyrecest.evaluation import generate_measurements
from shapely.geometry import Polygon


class TestGenerateMeasurementsEotFixedCounts(unittest.TestCase):
    def test_eot_accepts_numpy_array_fixed_measurement_counts(self):
        simulation_param = {
            "eot": True,
            "target_shape": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            "eot_sampling_style": "within",
            "n_timesteps": 2,
            "n_meas_at_individual_time_step": np.array([2, 3]),
        }
        groundtruth = np.zeros((2, 2))

        measurements = generate_measurements(groundtruth, simulation_param)

        self.assertEqual(measurements[0].shape, (2, 2))
        self.assertEqual(measurements[1].shape, (3, 2))

    def test_eot_normalizes_integral_float_fixed_measurement_counts(self):
        simulation_param = {
            "eot": True,
            "target_shape": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            "eot_sampling_style": "within",
            "n_timesteps": 2,
            "n_meas_at_individual_time_step": np.array([2.0, 3.0]),
        }
        groundtruth = np.zeros((2, 2))

        measurements = generate_measurements(groundtruth, simulation_param)

        self.assertEqual(measurements[0].shape, (2, 2))
        self.assertEqual(measurements[1].shape, (3, 2))

    def test_eot_rejects_invalid_fixed_measurement_counts(self):
        invalid_counts = (
            np.array([1.5]),
            np.array([-1]),
            np.array([np.nan]),
            np.array([True]),
        )

        for counts in invalid_counts:
            with self.subTest(counts=counts):
                simulation_param = {
                    "eot": True,
                    "target_shape": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    "eot_sampling_style": "within",
                    "n_timesteps": 1,
                    "n_meas_at_individual_time_step": counts,
                }
                groundtruth = np.zeros((1, 2))

                with self.assertRaisesRegex(
                    ValueError,
                    "n_meas_at_individual_time_step\\[0\\] must be a non-negative integer",
                ):
                    generate_measurements(groundtruth, simulation_param)


if __name__ == "__main__":
    unittest.main()
