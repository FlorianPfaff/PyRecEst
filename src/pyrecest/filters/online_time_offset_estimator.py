"""Online scalar timestamp-offset estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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

    def predict(self, dt: float = 0.0) -> None:
        """Apply random-walk process noise."""
        del dt
        self.variance = float(max(self.variance + self.process_variance, 1.0e-12))

    def update_from_position_residual(
        self,
        *,
        residual: np.ndarray,
        velocity: np.ndarray,
        measurement_variance: float,
    ) -> float:
        """Update the offset from a position residual and return innovation NIS."""
        residual = np.asarray(residual, dtype=float).reshape(-1)
        velocity = np.asarray(velocity, dtype=float).reshape(-1)
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


def _as_finite_scalar(value: Any, name: str) -> float:
    value_array = np.asarray(value)
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
