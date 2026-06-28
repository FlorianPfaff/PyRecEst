import pytest
from pyrecest import scenarios


@pytest.mark.parametrize(
    "scenario_type",
    [None, "", "   ", b"custom", 1, ["custom"]],
)
def test_register_scenario_runner_rejects_invalid_scenario_types(scenario_type):
    with pytest.raises(ValueError, match="scenario_type must be a non-empty string"):
        scenarios.register_scenario_runner(scenario_type, lambda _path: None)


def test_scenario_runner_decorator_rejects_invalid_scenario_type():
    with pytest.raises(ValueError, match="scenario_type must be a non-empty string"):
        scenarios.scenario_runner("   ")


def test_register_scenario_runner_strips_whitespace():
    name = "temporary_scenario_type_for_validation"
    scenarios._SCENARIO_RUNNERS.pop(name, None)  # pylint: disable=protected-access

    def runner(_path):
        raise AssertionError("runner should not be called")

    try:
        assert scenarios.register_scenario_runner(f"  {name}\t", runner) is runner
        assert name in scenarios.available_scenario_types()
        assert (
            f"  {name}\t" not in scenarios._SCENARIO_RUNNERS
        )  # pylint: disable=protected-access

        with pytest.raises(ValueError, match="already registered"):
            scenarios.register_scenario_runner(name, runner)
    finally:
        scenarios._SCENARIO_RUNNERS.pop(name, None)  # pylint: disable=protected-access


def test_run_scenario_rejects_unhashable_scenario_type(tmp_path):
    scenario_path = tmp_path / "bad_scenario.toml"
    scenario_path.write_text(
        "[scenario]\n" 'type = ["linear_gaussian"]\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported scenario type"):
        scenarios.run_scenario(scenario_path)
