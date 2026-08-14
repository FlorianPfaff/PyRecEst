import math
from pathlib import Path

from pyrecest.scenarios import run_scenario


def test_particle_resampling_scenario_uses_overflow_safe_sample_mean_norm(
    tmp_path: Path,
):
    scenario = tmp_path / "extreme_particle_norm.toml"
    scenario.write_text(
        """
[scenario]
type = "particle_resampling"
name = "extreme-particle-norm"
seed = 0

[data]
particles = [[1e200, 1e200]]
weights = [1.0]
num_samples = 1
""".strip(),
        encoding="utf-8",
    )

    result = run_scenario(scenario)

    assert math.isfinite(result.metrics["sample_mean_norm"])
    assert result.metrics["sample_mean_norm"] == math.hypot(1e200, 1e200)
    assert "Infinity" not in result.to_json()
