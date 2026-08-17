import pytest
from pyrecest.backend import eye, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.evaluation.check_and_fix_config import check_and_fix_config


def test_check_and_fix_config_rejects_none_timesteps_before_derived_fields():
    simulation_param = {
        "n_timesteps": None,
        "initial_prior": GaussianDistribution(zeros(1), eye(1)),
    }

    with pytest.raises(TypeError, match="n_timesteps must be an integer"):
        check_and_fix_config(simulation_param)
