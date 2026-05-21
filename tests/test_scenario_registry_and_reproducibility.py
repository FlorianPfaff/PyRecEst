from pathlib import Path

from pyrecest.scenarios import available_scenario_types, run_scenario


def test_particle_resampling_scenario_is_seed_reproducible(tmp_path: Path):
    scenario = tmp_path / "particle.toml"
    scenario.write_text(
        """
[scenario]
type = "particle_resampling"
name = "seeded-particle-resampling"
seed = 7

[data]
particles = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
weights = [0.1, 0.2, 0.7]
num_samples = 8
""".strip(),
        encoding="utf-8",
    )

    first = run_scenario(scenario)
    second = run_scenario(scenario)

    assert "particle_resampling" in available_scenario_types()
    assert first.final_estimate == second.final_estimate
    assert (
        first.diagnostics["metadata"]["indices"]
        == second.diagnostics["metadata"]["indices"]
    )
    assert first.metrics["effective_sample_size"] > 0.0
