"""Regression tests for missing simulation horizon validation."""

import pytest
from pyrecest.evaluation import evaluate_for_simulation_config


def test_custom_scenario_rejects_missing_timesteps_before_generation():
    """The custom scenario database entry stores a present-but-None horizon."""
    with pytest.warns(UserWarning, match="Scenario not recognized"):
        with pytest.raises(ValueError, match="n_steps must be provided"):
            evaluate_for_simulation_config(
                "custom",
                [],
                n_runs=1,
                scenario_customization_params={},
            )


def test_explicit_none_timesteps_in_mapping_is_rejected():
    with pytest.raises(ValueError, match="n_steps must be provided"):
        evaluate_for_simulation_config(
            {"n_timesteps": None},
            [],
            n_runs=1,
        )
