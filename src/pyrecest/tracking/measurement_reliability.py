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
from pyrecest.numerics import assert_covariance_matrix

ReliabilityMode = Literal["off", "inflate", "hard"] | str
_TEXT_OR_BOOL_SCALAR_TYPES = (
    bool,
    np.bool_,
    str,
    bytes,
    bytearray,
    np.str_,
    np.bytes_,
)
_COMPLEX_SCALAR_TYPES = (complex, np.complexfloating)
_REJECTED_NUMERIC_ARRAY_KINDS = frozenset({"b", "c", "S", "U", "M", "m"})


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
            else _scale_upper_bound(self.max_scale, "max_scale")
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
        object.__setattr__(self, "accepted", _bool_scalar(self.accepted, "accepted"))
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
        measurement = _real_numeric_array(self.measurement, "measurement").reshape(-1)
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
    bounded_scale = (
        None if max_scale is None else _scale_upper_bound(max_scale, "max_scale")
    )
    effective_reliability = max(reliability, floor)
    log_scale = -exponent * np.log(effective_reliability)
    if bounded_scale is not None and log_scale >= np.log(bounded_scale):
        return float(bounded_scale)
    if log_scale > np.log(np.finfo(float).max):
        raise ValueError(
            "covariance scale overflows; set max_scale to a finite upper bound"
        )
    scale = 1.0 / effective_reliability**exponent
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


def _scale_upper_bound(value: float, name: str) -> float:
    value = _positive_scalar(value, name)
    if value < 1.0:
        raise ValueError(f"{name} must be at least 1")
    return value


def _finite_scalar(value: Any, name: str) -> float:
    array = np.asarray(value)
    if array.shape != () or array.dtype == np.bool_ or array.dtype.kind == "c":
        raise ValueError(f"{name} must be a finite scalar")
    scalar = array.item()
    if isinstance(scalar, _TEXT_OR_BOOL_SCALAR_TYPES + _COMPLEX_SCALAR_TYPES):
        raise ValueError(f"{name} must be a finite scalar")
    try:
        parsed = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _real_numeric_array(value: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind in _REJECTED_NUMERIC_ARRAY_KINDS:
        raise ValueError(f"{name} must contain real numeric values")
    if raw.dtype == object:
        for item in raw.ravel():
            if isinstance(item, _TEXT_OR_BOOL_SCALAR_TYPES) or isinstance(
                item, (complex, np.complexfloating)
            ):
                raise ValueError(f"{name} must contain real numeric values")
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc


def _bool_scalar(value: Any, name: str) -> bool:
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean")
    return bool(array.item())


def _covariance_matrix(value: Any, *, dim: int | None = None) -> np.ndarray:
    covariance = _real_numeric_array(value, "covariance")
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if dim is not None and covariance.shape != (dim, dim):
        raise ValueError(f"covariance must have shape ({dim}, {dim})")
    if not np.isfinite(covariance).all():
        raise ValueError("covariance must contain only finite values")
    assert_covariance_matrix(covariance, name="covariance")
    return covariance.copy()


__all__ = [
    "MeasurementReliabilityConfig",
    "MeasurementReliabilityResult",
    "ReliabilityWeightedMeasurement",
    "apply_measurement_reliability",
    "reliability_to_covariance_scale",
    "scale_covariance_by_reliability",
]
