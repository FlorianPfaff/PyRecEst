import unittest

import numpy as np

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, random
from pyrecest.distributions import GaussianDistribution
from pyrecest.evaluation import generate_measurements


class TestGenerateMeasurementsShape(unittest.TestCase):
    def test_single_gaussian_measurement_keeps_measurement_axis(self):
        random.seed(0)
        simulation_config = {
            "n_timesteps": 2,
            "n_meas_at_individual_time_step": np.ones(2, dtype=int),
            "meas_noise": GaussianDistribution(
                array([0.0, 0.0]),
                array([[1e-24, 0.0], [0.0, 1e-24]]),
                check_validity=False,
            ),
        }
        groundtruth = array([[1.0, 2.0], [3.0, 4.0]])

        measurements = generate_measurements(groundtruth, simulation_config)

        self.assertEqual(tuple(measurements[0].shape), (1, 2))
        self.assertEqual(tuple(measurements[1].shape), (1, 2))


if __name__ == "__main__":
    unittest.main()
