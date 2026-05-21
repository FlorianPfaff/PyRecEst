import json
from pathlib import Path

from pyrecest.scenarios import load_scenario_config, run_scenario

SCENARIO = Path("scenarios/linear_gaussian_cv_1d/config.toml")
EXPECTED = Path("scenarios/linear_gaussian_cv_1d/expected.json")


def test_load_scenario_config():
    config = load_scenario_config(SCENARIO)
    assert config["scenario"]["type"] == "linear_gaussian"
    assert config["data"]["measurements"]


def test_linear_gaussian_scenario_matches_golden_output():
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    result = run_scenario(SCENARIO)
    tolerance = float(expected["tolerance"])
    errors = [
        abs(a - b) for a, b in zip(result.final_estimate, expected["final_estimate"])
    ]
    assert max(errors) <= tolerance
    assert result.metrics["max_abs_final_estimate_error"] <= tolerance
