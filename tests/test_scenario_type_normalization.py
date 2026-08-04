from pathlib import Path

from pyrecest.scenarios import run_scenario


def test_linear_gaussian_scenario_type_uses_dispatch_normalization(tmp_path: Path):
    scenario = tmp_path / "linear.toml"
    scenario.write_text(
        """
[scenario]
type = "  linear_gaussian  "
name = "normalized-linear"

[model]
system_matrix = [[1.0]]
system_noise_covariance = [[0.0]]

[measurement]
measurement_matrix = [[1.0]]
measurement_noise_covariance = [[1.0]]

[initial]
mean = [0.0]
covariance = [[1.0]]

[data]
measurements = []
""".strip(),
        encoding="utf-8",
    )

    result = run_scenario(scenario)

    assert result.name == "normalized-linear"
    assert result.final_estimate == [0.0]


def test_particle_resampling_scenario_type_uses_dispatch_normalization(
    tmp_path: Path,
):
    scenario = tmp_path / "particle.toml"
    scenario.write_text(
        """
[scenario]
type = "  particle_resampling  "
name = "normalized-particle"
seed = 7

[data]
particles = [[0.0], [1.0]]
weights = [0.5, 0.5]
num_samples = 2
""".strip(),
        encoding="utf-8",
    )

    result = run_scenario(scenario)

    assert result.name == "normalized-particle"
    assert len(result.estimates) == 2
