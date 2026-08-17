import numpy as np
import pytest
from pyrecest.backend import array, eye, get_backend_name
from pyrecest.distributions import GaussianDistribution
from pyrecest.evaluation import generate_measurements

pytestmark = pytest.mark.skipif(
    get_backend_name() == "jax",
    reason="MTT measurement generation is unsupported with JAX.",
)


def test_mtt_rejects_measurement_noise_dimension_mismatch():
    simulation_config = {
        "mtt": True,
        "eot": False,
        "n_timesteps": 1,
        "n_targets": 1,
        "clutter_rate": 0,
        "detection_probability": 1.0,
        "meas_matrix_for_each_target": eye(2),
        "meas_noise": GaussianDistribution(array([0.0]), eye(1)),
    }
    groundtruth = np.array([[[1.0, 2.0]]])

    with pytest.raises(
        ValueError,
        match=r"meas_noise\.sample\(1\).*got \(1, 1\)",
    ):
        generate_measurements(groundtruth, simulation_config)
