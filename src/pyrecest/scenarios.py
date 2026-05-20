"""Scenario loading and execution helpers for reproducible examples."""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ScenarioResult:
    """Serializable result returned by scenario runners."""

    name: str
    backend: str
    final_estimate: list[float]
    estimates: list[list[float]]
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def load_scenario_config(path: str | Path) -> dict[str, Any]:
    """Load a TOML scenario configuration."""
    scenario_path = Path(path)
    with scenario_path.open("rb") as handle:
        return tomllib.load(handle)


def _to_float_list(value: Any) -> list[float]:
    try:
        from pyrecest.backend import to_numpy

        value = to_numpy(value)
    except Exception:  # pragma: no cover - fallback for backend-specific values
        pass

    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def run_linear_gaussian_scenario(path: str | Path) -> ScenarioResult:
    """Run a constant-size linear Gaussian Kalman filtering scenario.

    The TOML format intentionally mirrors the mathematical notation used in
    the quickstart: transition matrix ``F``, measurement matrix ``H``, process
    covariance ``Q``, and measurement covariance ``R``.
    """
    config = load_scenario_config(path)
    if config.get("scenario", {}).get("type") != "linear_gaussian":
        raise ValueError("Only scenario.type = 'linear_gaussian' is supported by this runner.")

    from pyrecest import backend as be
    from pyrecest.filters import KalmanFilter

    model = config["model"]
    measurement = config["measurement"]
    initial = config["initial"]
    data = config["data"]

    system_matrix = be.array(model["system_matrix"])
    system_noise_cov = be.array(model["system_noise_covariance"])
    measurement_matrix = be.array(measurement["measurement_matrix"])
    measurement_noise_cov = be.array(measurement["measurement_noise_covariance"])

    kalman_filter = KalmanFilter(
        (
            be.array(initial["mean"]),
            be.array(initial["covariance"]),
        )
    )

    estimates: list[list[float]] = []
    for scalar_measurement in data["measurements"]:
        kalman_filter.predict_linear(system_matrix, system_noise_cov)
        kalman_filter.update_linear(
            be.array([float(scalar_measurement)]),
            measurement_matrix,
            measurement_noise_cov,
        )
        estimates.append(_to_float_list(kalman_filter.get_point_estimate()))

    final_estimate = estimates[-1] if estimates else _to_float_list(kalman_filter.get_point_estimate())
    expected = config.get("expected", {})
    metrics: dict[str, float] = {}
    if "final_estimate" in expected and final_estimate:
        errors = [abs(a - float(b)) for a, b in zip(final_estimate, expected["final_estimate"])]
        metrics["max_abs_final_estimate_error"] = max(errors) if errors else 0.0

    return ScenarioResult(
        name=config.get("scenario", {}).get("name", Path(path).stem),
        backend=getattr(be, "__backend_name__", "unknown"),
        final_estimate=final_estimate,
        estimates=estimates,
        metrics=metrics,
    )


def run_scenario(path: str | Path) -> ScenarioResult:
    """Run the scenario described by ``path``."""
    config = load_scenario_config(path)
    scenario_type = config.get("scenario", {}).get("type")
    if scenario_type == "linear_gaussian":
        return run_linear_gaussian_scenario(path)
    raise ValueError(f"Unsupported scenario type: {scenario_type!r}")


__all__ = [
    "ScenarioResult",
    "load_scenario_config",
    "run_linear_gaussian_scenario",
    "run_scenario",
]
