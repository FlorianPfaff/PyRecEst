import numpy as np
import pytest
from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.evaluation import check_and_fix_config


def _mtt_config(detection_probability):
    return {
        "mtt": True,
        "eot": False,
        "n_timesteps": 2,
        "n_targets": 1,
        "initial_prior": GaussianDistribution(array([0.0]), eye(1)),
        "meas_matrix_for_each_target": eye(1),
        "meas_noise": GaussianDistribution(array([0.0]), eye(1)),
        "detection_probability": detection_probability,
    }


@pytest.mark.parametrize("probability", [-0.1, 1.1, np.nan, np.inf, -np.inf])
def test_rejects_out_of_range_or_nonfinite_detection_probability(probability):
    with pytest.raises(
        ValueError,
        match="detection_probability must be finite and between 0 and 1",
    ):
        check_and_fix_config(_mtt_config(probability))


@pytest.mark.parametrize("probability", [True, False, "0.5", [0.5]])
def test_rejects_non_real_detection_probability(probability):
    with pytest.raises(TypeError, match="detection_probability must be a real scalar"):
        check_and_fix_config(_mtt_config(probability))


@pytest.mark.parametrize("probability", [0, 0.25, np.float32(0.5), 1])
def test_accepts_valid_detection_probability(probability):
    config = check_and_fix_config(_mtt_config(probability))

    assert config["detection_probability"] == pytest.approx(float(probability))
    assert isinstance(config["detection_probability"], float)
