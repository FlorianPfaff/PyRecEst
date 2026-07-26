import pytest
from pyrecest import scenarios


@pytest.mark.parametrize(
    "scenario_value",
    ['"linear_gaussian"', "1", '["linear_gaussian"]'],
)
@pytest.mark.parametrize(
    "runner",
    [
        scenarios.run_scenario,
        scenarios.run_linear_gaussian_scenario,
        scenarios.run_particle_resampling_scenario,
    ],
)
def test_scenario_entry_points_reject_non_table_scenario_section(
    tmp_path,
    scenario_value,
    runner,
):
    scenario_path = tmp_path / "bad_scenario.toml"
    scenario_path.write_text(f"scenario = {scenario_value}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="scenario must be a TOML table"):
        runner(scenario_path)
