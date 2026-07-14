import math
import sys
from pathlib import Path

from pyrecest.scenarios import run_scenario


def test_particle_resampling_scenario_normalizes_extreme_finite_weights(
    tmp_path: Path,
):
    max_float = sys.float_info.max
    scenario = tmp_path / "extreme_particle_weights.toml"
    scenario.write_text(
        f"""
[scenario]
type = "particle_resampling"
name = "extreme-particle-weights"
seed = 7

[data]
particles = [[0.0], [1.0]]
weights = [{max_float!r}, {max_float / 2.0!r}]
num_samples = 8
""".strip(),
        encoding="utf-8",
    )

    result = run_scenario(scenario)

    assert math.isclose(result.metrics["max_weight"], 2.0 / 3.0)
    assert math.isclose(result.metrics["min_weight"], 1.0 / 3.0)
    assert math.isclose(
        result.metrics["max_weight"] + result.metrics["min_weight"],
        1.0,
    )
    assert math.isfinite(result.metrics["effective_sample_size"])
