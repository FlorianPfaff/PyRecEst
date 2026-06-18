import pytest
from pyrecest.evaluation.simulation_database import simulation_database


def test_custom_scenario_requires_customization_parameters():
    with pytest.warns(UserWarning, match="Scenario not recognized"):
        with pytest.raises(ValueError, match="scenario_customization_params"):
            simulation_database("custom")
