import json
from pathlib import Path

import pytest
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
    assert len(result.final_estimate) == len(expected["final_estimate"])
    errors = [
        abs(a - b) for a, b in zip(result.final_estimate, expected["final_estimate"])
    ]
    assert max(errors) <= tolerance
    assert result.metrics["max_abs_final_estimate_error"] <= tolerance


def test_linear_gaussian_scenario_accepts_vector_measurements(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[scenario]
type = "linear_gaussian"
name = "vector measurement scenario"

[model]
system_matrix = [[1.0, 0.0], [0.0, 1.0]]
system_noise_covariance = [[0.0, 0.0], [0.0, 0.0]]

[measurement]
measurement_matrix = [[1.0, 0.0], [0.0, 1.0]]
measurement_noise_covariance = [[1.0, 0.0], [0.0, 1.0]]

[initial]
mean = [0.0, 0.0]
covariance = [[1.0, 0.0], [0.0, 1.0]]

[data]
measurements = [[1.0, 2.0], [3.0, 4.0]]
""".strip(),
        encoding="utf-8",
    )

    result = run_scenario(config_path)

    assert result.name == "vector measurement scenario"
    assert len(result.final_estimate) == 2
    assert len(result.estimates) == 2
    assert all(len(estimate) == 2 for estimate in result.estimates)


def test_linear_gaussian_scenario_rejects_expected_length_mismatch(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[scenario]
type = "linear_gaussian"
name = "expected length mismatch scenario"

[model]
system_matrix = [[1.0, 0.0], [0.0, 1.0]]
system_noise_covariance = [[0.0, 0.0], [0.0, 0.0]]

[measurement]
measurement_matrix = [[1.0, 0.0], [0.0, 1.0]]
measurement_noise_covariance = [[1.0, 0.0], [0.0, 1.0]]

[initial]
mean = [0.0, 0.0]
covariance = [[1.0, 0.0], [0.0, 1.0]]

[data]
measurements = [[1.0, 2.0]]

[expected]
final_estimate = [1.0]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="expected.final_estimate must have the same length",
    ):
        run_scenario(config_path)
