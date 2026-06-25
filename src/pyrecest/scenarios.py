"""Scenario loading and execution helpers for reproducible examples."""

from __future__ import annotations

import json
import math
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_TEXT_TYPES = (str, bytes, bytearray)


@dataclass(slots=True)
class ScenarioResult:
    """Serializable result returned by scenario runners."""

    name: str
    backend: str
    final_estimate: list[float]
    estimates: list[list[float]]
    metrics: dict[str, float]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


ScenarioRunner = Callable[[str | Path], ScenarioResult]
_SCENARIO_RUNNERS: dict[str, ScenarioRunner] = {}


def register_scenario_runner(
    scenario_type: str, runner: ScenarioRunner
) -> ScenarioRunner:
    """Register ``runner`` for a TOML ``scenario.type`` value."""
    if not scenario_type:
        raise ValueError("scenario_type must be a non-empty string")
    if scenario_type in _SCENARIO_RUNNERS:
        raise ValueError(f"Scenario type {scenario_type!r} is already registered")
    _SCENARIO_RUNNERS[scenario_type] = runner
    return runner


def scenario_runner(scenario_type: str) -> Callable[[ScenarioRunner], ScenarioRunner]:
    """Decorator form of :func:`register_scenario_runner`."""

    def decorator(runner: ScenarioRunner) -> ScenarioRunner:
        return register_scenario_runner(scenario_type, runner)

    return decorator


def available_scenario_types() -> tuple[str, ...]:
    """Return registered scenario types in a stable order."""
    return tuple(sorted(_SCENARIO_RUNNERS))


def load_scenario_config(path: str | Path) -> dict[str, Any]:
    """Load a TOML scenario configuration."""
    scenario_path = Path(path)
    with scenario_path.open("rb") as handle:
        return tomllib.load(handle)


def _numeric_config_scalar(value: Any, message: str) -> float:
    if isinstance(value, bool) or isinstance(value, _TEXT_TYPES):
        raise ValueError(message)
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc


def _integer_config_value(value: Any, name: str, *, positive: bool) -> int:
    descriptor = "positive" if positive else "non-negative"
    message = f"{name} must be a {descriptor} integer"
    scalar = _numeric_config_scalar(value, message)
    if not math.isfinite(scalar) or not scalar.is_integer():
        raise ValueError(message)
    integer = int(scalar)
    if positive:
        if integer <= 0:
            raise ValueError(message)
    elif integer < 0:
        raise ValueError(message)
    return integer


def _scenario_seed(config: dict[str, Any]) -> int | None:
    for section_name in ("random", "scenario"):
        section = config.get(section_name, {})
        if isinstance(section, dict) and "seed" in section:
            return _integer_config_value(section["seed"], "seed", positive=False)
    if "seed" in config:
        return _integer_config_value(config["seed"], "seed", positive=False)
    return None


def _apply_scenario_seed(config: dict[str, Any]) -> int | None:
    seed = _scenario_seed(config)
    if seed is None:
        return None
    from pyrecest.reproducibility import seed_all

    seed_all(seed)
    return seed


def _to_float_list(
    value: Any,
    *,
    name: str = "value",
    reject_text_or_bool: bool = False,
) -> list[float]:
    try:
        from pyrecest.backend import to_numpy

        value = to_numpy(value)
    except Exception:  # pragma: no cover  # pylint: disable=broad-exception-caught
        pass

    if hasattr(value, "tolist"):
        value = value.tolist()

    message = f"{name} must contain numeric values"

    def convert(item: Any) -> float:
        if reject_text_or_bool and (
            isinstance(item, bool) or isinstance(item, _TEXT_TYPES)
        ):
            raise ValueError(message)
        try:
            return float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(message) from exc

    if reject_text_or_bool and isinstance(value, _TEXT_TYPES):
        raise ValueError(message)
    if isinstance(value, int | float):
        return [convert(value)]
    return [convert(item) for item in value]


def _normalized_particle_weights(raw_weights: Any, particle_count: int, backend):
    if particle_count <= 0:
        raise ValueError("particle_resampling scenarios require at least one particle")

    if raw_weights is None:
        weight_values = [1.0 for _ in range(particle_count)]
    else:
        weight_values = _to_float_list(
            raw_weights,
            name="weights",
            reject_text_or_bool=True,
        )

    if len(weight_values) != particle_count:
        raise ValueError("weights must contain one entry per particle")
    if any(not math.isfinite(weight) for weight in weight_values):
        raise ValueError("weights must be finite")
    if any(weight < 0.0 for weight in weight_values):
        raise ValueError("weights must be nonnegative")

    weight_total = sum(weight_values)
    if weight_total <= 0.0:
        raise ValueError("weights must have positive total mass")

    return backend.asarray(
        [weight / weight_total for weight in weight_values],
        dtype=backend.float64,
    )


def _positive_integer_config_value(value: Any, name: str) -> int:
    return _integer_config_value(value, name, positive=True)


@scenario_runner("linear_gaussian")
def run_linear_gaussian_scenario(path: str | Path) -> ScenarioResult:
    """Run a constant-size linear Gaussian Kalman filtering scenario.

    The TOML format intentionally mirrors the mathematical notation used in
    the quickstart: transition matrix ``F``, measurement matrix ``H``, process
    covariance ``Q``, and measurement covariance ``R``.
    """
    config = load_scenario_config(path)
    if config.get("scenario", {}).get("type") != "linear_gaussian":
        raise ValueError(
            "Only scenario.type = 'linear_gaussian' is supported by this runner."
        )
    seed = _apply_scenario_seed(config)

    from pyrecest import backend as be
    from pyrecest.filters.kalman_filter import KalmanFilter

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
    nis_values: list[float] = []
    for scalar_measurement in data["measurements"]:
        kalman_filter.predict_linear(system_matrix, system_noise_cov)
        diagnostics = kalman_filter.update_linear(
            be.array([float(scalar_measurement)]),
            measurement_matrix,
            measurement_noise_cov,
            return_diagnostics=True,
        )
        if diagnostics is not None and diagnostics.get("nis") is not None:
            nis_values.append(float(diagnostics["nis"]))
        estimates.append(_to_float_list(kalman_filter.get_point_estimate()))

    final_estimate = (
        estimates[-1]
        if estimates
        else _to_float_list(kalman_filter.get_point_estimate())
    )
    expected = config.get("expected", {})
    metrics: dict[str, float] = {}
    if "final_estimate" in expected and final_estimate:
        errors = [
            abs(a - float(b))
            for a, b in zip(final_estimate, expected["final_estimate"])
        ]
        metrics["max_abs_final_estimate_error"] = max(errors) if errors else 0.0
    if nis_values:
        metrics["mean_nis"] = sum(nis_values) / len(nis_values)
        metrics["max_nis"] = max(nis_values)

    return ScenarioResult(
        name=config.get("scenario", {}).get("name", Path(path).stem),
        backend=getattr(be, "__backend_name__", "unknown"),
        final_estimate=final_estimate,
        estimates=estimates,
        metrics=metrics,
        diagnostics={"seed": seed, "nis": nis_values},
    )


@scenario_runner("particle_resampling")
def run_particle_resampling_scenario(path: str | Path) -> ScenarioResult:
    """Run a deterministic weighted-resampling smoke scenario.

    This compact scenario is useful for CI because it exercises backend random
    state, particle weights, and serializable diagnostics without depending on a
    large tracker configuration.
    """
    config = load_scenario_config(path)
    if config.get("scenario", {}).get("type") != "particle_resampling":
        raise ValueError(
            "Only scenario.type = 'particle_resampling' is supported by this runner."
        )
    seed = _apply_scenario_seed(config)

    from pyrecest import backend as be
    from pyrecest.diagnostics import ParticleDiagnostics

    data = config["data"]
    particles = be.asarray(data["particles"], dtype=be.float64)
    weights = _normalized_particle_weights(
        data.get("weights"),
        int(particles.shape[0]),
        be,
    )
    num_samples = _positive_integer_config_value(
        data.get("num_samples", int(particles.shape[0])),
        "num_samples",
    )

    indices = be.random.choice(
        be.arange(int(particles.shape[0])),
        size=num_samples,
        replace=True,
        p=weights,
    )
    sampled_particles = particles[indices]
    sampled_mean = be.sum(sampled_particles, axis=0) / float(num_samples)
    final_estimate = _to_float_list(sampled_mean)

    weights_py = _to_float_list(weights)
    diagnostics = ParticleDiagnostics.from_weights(
        weights_py,
        resampled=True,
        resampling_count=1,
        metadata={
            "seed": seed,
            "indices": _to_float_list(indices),
        },
    )
    metrics = {
        "effective_sample_size": float(diagnostics.effective_sample_size or 0.0),
        "weight_entropy": float(diagnostics.weight_entropy or 0.0),
        "max_weight": max(weights_py),
        "min_weight": min(weights_py),
        "sample_mean_norm": math.sqrt(sum(value * value for value in final_estimate)),
    }

    return ScenarioResult(
        name=config.get("scenario", {}).get("name", Path(path).stem),
        backend=getattr(be, "__backend_name__", "unknown"),
        final_estimate=final_estimate,
        estimates=[_to_float_list(row) for row in sampled_particles],
        metrics=metrics,
        diagnostics=diagnostics.to_dict(),
    )


def run_scenario(path: str | Path) -> ScenarioResult:
    """Run the scenario described by ``path``."""
    config = load_scenario_config(path)
    scenario_type = config.get("scenario", {}).get("type")
    runner = _SCENARIO_RUNNERS.get(scenario_type)
    if runner is None:
        available = ", ".join(available_scenario_types()) or "none"
        raise ValueError(
            f"Unsupported scenario type: {scenario_type!r}. Available scenario types: {available}."
        )
    return runner(path)


__all__ = [
    "ScenarioResult",
    "available_scenario_types",
    "load_scenario_config",
    "register_scenario_runner",
    "run_linear_gaussian_scenario",
    "run_particle_resampling_scenario",
    "run_scenario",
    "scenario_runner",
]
