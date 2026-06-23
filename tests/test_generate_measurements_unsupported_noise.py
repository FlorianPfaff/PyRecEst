import unittest

import numpy as np
from pyrecest.evaluation.generate_measurements import generate_measurements


class TestGenerateMeasurementsUnsupportedNoise(unittest.TestCase):
    def test_rejects_unsupported_measurement_noise_distribution(self):
        simulation_config = {
            "n_timesteps": 2,
            "n_meas_at_individual_time_step": np.ones(2, dtype=int),
            "meas_noise": object(),
        }

        with self.assertRaisesRegex(TypeError, "meas_noise"):
            generate_measurements(np.zeros((2, 1)), simulation_config)


if __name__ == "__main__":
    unittest.main()
