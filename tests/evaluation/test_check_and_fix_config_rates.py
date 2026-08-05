import numpy as np
import pytest
from pyrecest.backend import eye, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.evaluation.check_and_fix_config import check_and_fix_config


def _base_config(**overrides):
    config = {
        "n_timesteps": 3,
        "initial_prior": GaussianDistribution(zeros(1), eye(1)),
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize(
    ("intensity_lambda", "expected_error"),
    [
        (np.nan, ValueError),
        (np.inf, ValueError),
        (True, TypeError),
    ],
)
def test_check_and_fix_config_rejects_invalid_intensity_rates(
    intensity_lambda, expected_error
):
    with pytest.raises(expected_error, match="Intensity lambda must be positive"):
        check_and_fix_config(
            _base_config(eot=True, intensity_lambda=intensity_lambda)
        )


@pytest.mark.parametrize(
    ("clutter_rate", "expected_error"),
    [
        (-1.0, ValueError),
        (np.nan, ValueError),
        (np.inf, ValueError),
        (True, TypeError),
    ],
)
def test_check_and_fix_config_rejects_invalid_clutter_rates(
    clutter_rate, expected_error
):
    with pytest.raises(expected_error, match="clutter_rate must be non-negative"):
        check_and_fix_config(
            _base_config(mtt=True, clutter_rate=clutter_rate, observed_area=1.0)
        )


def test_check_and_fix_config_normalizes_valid_event_rates():
    eot_config = check_and_fix_config(
        _base_config(eot=True, intensity_lambda=np.float64(2.5))
    )
    mtt_config = check_and_fix_config(
        _base_config(mtt=True, clutter_rate=np.float64(1.5), observed_area=1.0)
    )

    assert eot_config["intensity_lambda"] == 2.5
    assert isinstance(eot_config["intensity_lambda"], float)
    assert mtt_config["clutter_rate"] == 1.5
    assert isinstance(mtt_config["clutter_rate"], float)
