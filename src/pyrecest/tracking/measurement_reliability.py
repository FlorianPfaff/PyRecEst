"""Measurement reliability helpers for covariance scaling and hard rejection.

The utilities in this module are intentionally filter-independent.  A caller can
score a measurement with any truth-free reliability model, then use these helpers
to turn that score into a covariance inflation scale or a hard accept/reject
decision.  This is useful for asynchronous multi-sensor trackers where some
measurements are intermittent or have reliability metadata that should not be
encoded directly in the nominal Gaussian covariance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

ReliabilityMode = Literal["off", "inflate", "hard"] | str


@dataclass(frozen=True)
class MeasurementReliabilityConfig:
    """Configuration for converting reliability scores into update decisions."""

    mode: ReliabilityMode = "inflate"
    threshold: float | None = None
    floor: float = 0.05
    exponent: float = 1.0
    max_scale: float | None = None

    def __post_init__(self) -> None:
        mode = str(self.mode)
        if mode not in {"off", "inflate", "hard"}:
            raise ValueError("mode must be one of 'off', 'inflate', or 'hard'")
        threshold = (
            None
            if self.threshold is None
            else _probability(self.threshold, "threshold")
        )
        floor = _probability(self.floor, "floor")
        if floor <= 0.0:
            raise ValueError("floor must be positive")
        exponent = _positive_scalar(self.exponent, "exponent")
        max_scale = (
            None
            if self.max_scale is None
            else _positive_scalar(self.max_scale, "max_scale")
        )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "floor", floor)
        object.__setattr__(self, "exponent", exponent)
        object.__setattr__(self, "max_scale", max_scale)


@dataclass(frozen=True)
class MeasurementReliabilityResult:
    """Reliability-weighted measurement decision and covariance."""

    reliability: float
    covariance_scale: float
    covariance: np.ndarray
    accepted: bool
    action: str
    mode: str

    def __post_init__(self) -> None:
        covariance = _covariance_matrix(self.covariance)
        object.__setattr__(
            self, "reliability", _probability(self.reliability, "reliability")
        )
        object.__setattr__(
            self,
            "covariance_scale",
            _positive_scalar(self.covariance_scale, "covariance_scale"),
        )
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "action", str(self.action))
        object.__setattr__(self, "mode", str(self.mode))


@dataclass(frozen=True)
class ReliabilityWeightedMeasurement:
    """Container for a measurement with a truth-free reliability score."""

    measurement: np.ndarray
    covariance: np.ndarray
    reliability: float
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        measurement = np.asarray(self.measurement, dtype=float).reshape(-1)
        if measurement.size == 0:
            raise ValueError("measurement must contain at least one value")
        if not np.isfinite(measurement).all():
            raise ValueError("measurement must contain only finite values")
        covariance = _covariance_matrix(self.covariance, dim=measurement.size)
        object.__setattr__(self, "measurement", measurement.copy())
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(
            self, "reliability", _probability(self.reliability, "reliability")
        )
        object.__setattr__(
            self, "source", None if self.source is None else str(self.source)
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def apply_reliability(
        self,
        config: MeasurementReliabilityConfig | None = None,
        **kwargs: Any,
    ) -> MeasurementReliabilityResult:
        """Return the covariance/action decision for this measurement."""

        return apply_measurement_reliability(
            self.covariance,
            reliability=self.reliability,
            config=config,
            **kwargs,
        )


def reliability_to_covariance_scale(
    reliability: float,
    *,
    floor: float = 0.05,
    exponent: float = 1.0,
    max_scale: float | None = None,
) -> float:
    """Convert a probability-like reliability score into covariance scale.

    ``reliability=1`` leaves the covariance unchanged.  Lower reliability values
    inflate the covariance as ``1 / max(reliability, floor)**exponent``.  The
    floor prevents singularly large scales when a reliability model emits zero.
    """

    reliability = _probability(reliability, "reliability")
    floor = _probability(floor, "floor")
    if floor <= 0.0:
        raise ValueError("floor must be positive")
    exponent = _positive_scalar(exponent, "exponent")
    scale = 1.0 / max(reliability, floor) ** exponent
    if max_scale is not None:
        scale = min(scale, _positive_scalar(max_scale, "max_scale"))
    return float(max(1.0, scale))


def scale_covariance_by_reliability(
    covariance: np.ndarray,
    reliability: float,
    *,
    floor: float = 0.05,
    exponent: float = 1.0,
    max_scale: float | None = None,
) -> tuple[np.ndarray, float]:
    """Return ``covariance * scale`` and the applied reliability scale."""

    covariance = _covariance_matrix(covariance)
    scale = reliability_to_covariance_scale(
        reliability,
        floor=floor,
        exponent=exponent,
        max_scale=max_scale,
    )
    return covariance * scale, scale


def apply_measurement_reliability(
    covariance: np.ndarray,
    *,
    reliability: float,
    config: MeasurementReliabilityConfig | None = None,
    mode: ReliabilityMode | None = None,
    threshold: float | None = None,
    floor: float | None = None,
    exponent: float | None = None,
    max_scale: float | None = None,
) -> MeasurementReliabilityResult:
    """Apply reliability policy to one measurement covariance.

    Modes
    -----
    ``off``
        Always accept and return the nominal covariance.
    ``hard``
        Reject when ``reliability < threshold``. Accepted measurements keep the
        nominal covariance.
    ``inflate``
        Accept by default and inflate covariance by inverse reliability.  If a
        threshold is supplied, values below it are rejected before inflation.
    """

    base = MeasurementReliabilityConfig() if config is None else config
    effective = MeasurementReliabilityConfig(
        mode=base.mode if mode is None else mode,
        threshold=base.threshold if threshold is None else threshold,
        floor=base.floor if floor is None else floor,
        exponent=base.exponent if exponent is None else exponent,
        max_scale=base.max_scale if max_scale is None else max_scale,
    )
    covariance = _covariance_matrix(covariance)
    reliability = _probability(reliability, "reliability")

    if effective.mode == "off":
        return MeasurementReliabilityResult(
            reliability=reliability,
            covariance_scale=1.0,
            covariance=covariance,
            accepted=True,
            action="reliability_off",
            mode=effective.mode,
        )

    threshold_value = effective.threshold
    if effective.mode == "hard":
        if threshold_value is None:
            threshold_value = effective.floor
        accepted = reliability >= threshold_value
        return MeasurementReliabilityResult(
            reliability=reliability,
            covariance_scale=1.0,
            covariance=covariance,
            accepted=accepted,
            action="reliability_accepted" if accepted else "reliability_rejected",
            mode=effective.mode,
        )

    if threshold_value is not None and reliability < threshold_value:
        return MeasurementReliabilityResult(
            reliability=reliability,
            covariance_scale=1.0,
            covariance=covariance,
            accepted=False,
            action="reliability_rejected",
            mode=effective.mode,
        )

    scaled, scale = scale_covariance_by_reliability(
        covariance,
        reliability,
        floor=effective.floor,
        exponent=effective.exponent,
        max_scale=effective.max_scale,
    )
    return MeasurementReliabilityResult(
        reliability=reliability,
        covariance_scale=scale,
        covariance=scaled,
        accepted=True,
        action="reliability_inflated" if scale > 1.0 else "reliability_accepted",
        mode=effective.mode,
    )


def _probability(value: float, name: str) -> float:
    value = _finite_scalar(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _positive_scalar(value: float, name: str) -> float:
    value = _finite_scalar(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _finite_scalar(value: float, name: str) -> float:
    array = np.asarray(value)
    if array.shape != () or array.dtype == np.bool_:
        raise ValueError(f"{name} must be a finite scalar")
    parsed = float(array.item())
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _covariance_matrix(value: Any, *, dim: int | None = None) -> np.ndarray:
    covariance = np.asarray(value, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if dim is not None and covariance.shape != (dim, dim):
        raise ValueError(f"covariance must have shape ({dim}, {dim})")
    if not np.isfinite(covariance).all():
        raise ValueError("covariance must contain only finite values")
    return covariance.copy()


__all__ = [
    "MeasurementReliabilityConfig",
    "MeasurementReliabilityResult",
    "ReliabilityWeightedMeasurement",
    "apply_measurement_reliability",
    "reliability_to_covariance_scale",
    "scale_covariance_by_reliability",
]
