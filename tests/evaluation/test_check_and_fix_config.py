import pytest
from pyrecest.backend import eye, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.evaluation.check_and_fix_config import (
    _expand_meas_per_step,
    _validate_measurement_counts,
    check_and_fix_config,
)


def _base_config(**overrides):
    config = {
        "n_timesteps": 3,
        "initial_prior": GaussianDistribution(zeros(1), eye(1)),
    }
    config.update(overrides)
    return config


def test_expand_meas_per_step_uses_one_count_per_timestep():
    simulation_param = {"n_timesteps": 4, "meas_per_step": 3}

    _expand_meas_per_step(simulation_param)

    assert simulation_param["n_meas_at_individual_time_step"] == [3, 3, 3, 3]
    assert "meas_per_step" not in simulation_param


def test_expand_meas_per_step_rejects_nonpositive_count():
    simulation_param = {"n_timesteps": 4, "meas_per_step": 0}

    with pytest.raises(ValueError, match="positive"):
        _expand_meas_per_step(simulation_param)


def test_expand_meas_per_step_rejects_noninteger_count():
    simulation_param = {"n_timesteps": 4, "meas_per_step": 2.5}

    with pytest.raises(TypeError, match="integer"):
        _expand_meas_per_step(simulation_param)


def test_validate_measurement_counts_rejects_wrong_length():
    simulation_param = {
        "n_timesteps": 4,
        "n_meas_at_individual_time_step": [1, 1, 1],
    }

    with pytest.raises(ValueError, match="one entry per time step"):
        _validate_measurement_counts(simulation_param)


def test_validate_measurement_counts_rejects_noninteger_entries():
    simulation_param = {
        "n_timesteps": 2,
        "n_meas_at_individual_time_step": [1, 1.5],
    }

    with pytest.raises(TypeError, match="integer"):
        _validate_measurement_counts(simulation_param)


def test_validate_measurement_counts_rejects_nonpositive_entries():
    simulation_param = {
        "n_timesteps": 2,
        "n_meas_at_individual_time_step": [1, 0],
    }

    with pytest.raises(ValueError, match="positive"):
        _validate_measurement_counts(simulation_param)


def test_validate_measurement_counts_accepts_matching_length():
    simulation_param = {
        "n_timesteps": 3,
        "n_meas_at_individual_time_step": [1, 2, 3],
    }

    _validate_measurement_counts(simulation_param)


def test_check_and_fix_config_rejects_nonpositive_timesteps():
    with pytest.raises(ValueError, match="n_timesteps must be positive"):
        check_and_fix_config(_base_config(n_timesteps=0))


def test_check_and_fix_config_rejects_intensity_without_eot_or_mtt():
    with pytest.raises(ValueError, match="MTT or EOT"):
        check_and_fix_config(_base_config(intensity_lambda=1.0))


def test_check_and_fix_config_rejects_nonpositive_intensity():
    with pytest.raises(ValueError, match="Intensity lambda must be positive"):
        check_and_fix_config(_base_config(eot=True, intensity_lambda=0.0))


def test_check_and_fix_config_requires_one_eot_measurement_modulator():
    with pytest.raises(ValueError, match="precisely one"):
        check_and_fix_config(_base_config(eot=True))


def test_check_and_fix_config_rejects_mtt_measurement_count_fields():
    with pytest.raises(ValueError, match="MTT scenarios"):
        check_and_fix_config(
            _base_config(mtt=True, n_meas_at_individual_time_step=[1, 1, 1])
        )


def test_check_and_fix_config_rejects_clutter_without_observed_area():
    with pytest.raises(ValueError, match="observed_area"):
        check_and_fix_config(_base_config(mtt=True, clutter_rate=1.0))


def test_check_and_fix_config_rejects_invalid_initial_prior():
    with pytest.raises(TypeError, match="initial_prior"):
        check_and_fix_config(_base_config(initial_prior=object()))
