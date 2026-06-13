import pytest
from pyrecest.evaluation.check_and_fix_config import (
    _expand_meas_per_step,
    _validate_measurement_counts,
)


def test_expand_meas_per_step_uses_one_count_per_timestep():
    simulation_param = {"n_timesteps": 4, "meas_per_step": 3}

    _expand_meas_per_step(simulation_param)

    assert simulation_param["n_meas_at_individual_time_step"] == [3, 3, 3, 3]
    assert "meas_per_step" not in simulation_param


def test_validate_measurement_counts_rejects_wrong_length():
    simulation_param = {
        "n_timesteps": 4,
        "n_meas_at_individual_time_step": [1, 1, 1],
    }

    with pytest.raises(AssertionError, match="one entry per time step"):
        _validate_measurement_counts(simulation_param)


def test_validate_measurement_counts_accepts_matching_length():
    simulation_param = {
        "n_timesteps": 3,
        "n_meas_at_individual_time_step": [1, 2, 3],
    }

    _validate_measurement_counts(simulation_param)
