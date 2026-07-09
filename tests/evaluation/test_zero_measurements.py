import numpy as np
from pyrecest.distributions import GaussianDistribution
from pyrecest.evaluation.generate_measurements import generate_measurements


def test_zero_count_returns_empty_measurement_matrix():
    groundtruth = np.array([[1.0, 2.0]])
    config = {
        "n_timesteps": 1,
        "n_meas_at_individual_time_step": np.array([0]),
        "meas_noise": GaussianDistribution(np.zeros(2), np.eye(2)),
    }

    measurements = generate_measurements(groundtruth, config)

    assert measurements[0].shape == (0, 2)
