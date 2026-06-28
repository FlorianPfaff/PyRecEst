"""Online scalar timestamp-offset estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_UNSUPPORTED_NUMERIC_KINDS = {"b", "S", "U", "c", "M", "m"}
_UNSUPPORTED_SCALAR_TYPES = (
    type(None),
    bool,
    np.bool_,
    str,
    bytes,
    bytearray,
    np.str_,
    np.bytes_,
    complex,
    np.complexfloating,
    np.datetime64,
    np.timedelta64,
)


@dataclass
class OnlineTimeOffsetEstimator:
    """Scalar online timestamp-offset estimator with a Gaussian state."""

    offset: float = 0.0
    variance: float = 1.0
    process_variance: float = 1.0e-4
    min_speed: float = 1.0

    def __post_init__(self) -> None:
        self.offset = _as_finite_scalar(self.offset, "offset")
        self.variance = _as_finite_scalar(self.variance, "variance")
        self.process_variance = _as_finite_scalar(
            self.process_variance,
            "process_variance",
        )
        self.min_speed = _as_finite_scalar(self.min_speed, "min_speed")

        if self.variance <= 0.0:
            raise ValueError("variance must be positive")
        if self.process_variance < 0.0:
            raise ValueError("process_variance must be nonnegative")
        if self.min_speed < 0.0:
            raise ValueError("min_speed must be nonnegative")

    def predict(self, dt: float = 1.0) -> None:
        """Apply elapsed-time-scaled random-walk process noise."""
        dt_value = _as_finite_scalar(dt, "dt")
        if dt_value < 0.0:
            raise ValueError("dt must be nonnegative")
        self.variance = float(
            max(self.variance + self.process_variance * dt_value, 1.0e-12)
        )

    def update_from_position_residual(
        self,
        *,
        residual: np.ndarray,
        velocity: np.ndarray,
        measurement_variance: float,
    ) -> float:
        """Update the offset from a position residual and return innovation NIS."""
        residual = _as_real_numeric_array(residual, "residual").reshape(-1)
        velocity = _as_real_numeric_array(velocity, "velocity").reshape(-1)
        if residual.size != velocity.size:
            raise ValueError("residual and velocity must have the same dimension")
        if not np.isfinite(residual).all() or not np.isfinite(velocity).all():
            raise ValueError("residual and velocity must be finite")
        measurement_variance = _as_finite_scalar(
            measurement_variance,
            "measurement_variance",
        )
        if measurement_variance < 0.0:
            raise ValueError("measurement_variance must be nonnegative")

        speed2 = float(velocity @ velocity)
        if speed2 < float(self.min_speed) ** 2:
            return float("nan")
        measured_offset = float((residual @ velocity) / speed2)
        variance = max(float(measurement_variance) / speed2, 1.0e-12)
        innovation = measured_offset - float(self.offset)
        innovation_variance = float(self.variance + variance)
        gain = float(self.variance / innovation_variance)
        self.offset = float(self.offset + gain * innovation)
        self.variance = float(max((1.0 - gain) * self.variance, 1.0e-12))
        return float((innovation**2) / max(innovation_variance, 1.0e-12))

    @property
    def std(self) -> float:
        """Return the posterior offset standard deviation."""
        return float(np.sqrt(max(self.variance, 0.0)))


def _contains_unsupported_numeric_values(value: Any) -> bool:
    value_array = np.asarray(value)
    if value_array.dtype.kind in _UNSUPPORTED_NUMERIC_KINDS:
        return True
    if value_array.dtype.kind != "O":
        return False
    return any(isinstance(item, _UNSUPPORTED_SCALAR_TYPES) for item in value_array.flat)


def _as_real_numeric_array(value: Any, name: str) -> np.ndarray:
    try:
        if _contains_unsupported_numeric_values(value):
            raise ValueError
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be real-valued numeric") from exc


def _as_finite_scalar(value: Any, name: str) -> float:
    try:
        if _contains_unsupported_numeric_values(value):
            raise ValueError
        value_array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if value_array.shape != () or value_array.dtype == np.bool_:
        raise ValueError(f"{name} must be a finite scalar")
    scalar = value_array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite scalar")
    try:
        parsed = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be a finite scalar")
    return parsed
