from unittest.mock import Mock

import pyrecest.backend
import pytest
from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.evaluation import generate_groundtruth


pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
    reason="Groundtruth generation is unsupported on this backend.",
)


def test_rejects_system_noise_dimension_mismatch_before_broadcasting():
    noise = Mock()
    noise.sample.return_value = array([[0.5]])
    simulation_param = {
        "initial_prior": GaussianDistribution(array([0.0, 0.0]), eye(2)),
        "n_targets": 1,
        "n_timesteps": 2,
        "sys_noise": noise,
    }

    with pytest.raises(
        ValueError,
        match=r"sys_noise\.sample\(1\).*got \(1, 1\)",
    ):
        generate_groundtruth(simulation_param, array([1.0, 2.0]))
