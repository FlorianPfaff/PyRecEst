"""Adaptive process-noise scaling from innovation statistics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class AdaptiveProcessNoiseConfig:
    """Parameters for bounded NIS-ratio process-noise adaptation.

    ``base_scale`` is the nominal multiplier applied when the observed NIS per
    measurement dimension is near one.  The rolling ratio is mapped to a bounded
    scale using ``low_nis_ratio``, ``high_nis_ratio``, and ``scale_gain``.
    """

    base_scale: float = 1.0
    min_scale: float = 0.35
    max_scale: float = 4.0
    ewma_alpha: float = 0.05
    high_nis_ratio: float = 1.5
    low_nis_ratio: float = 0.6
    scale_gain: float = 0.5

    def __post_init__(self) -> None:
        base_scale = _normalize_finite_scalar(self.base_scale, "base_scale")
        min_scale = _normalize_finite_scalar(self.min_scale, "min_scale")
        max_scale = _normalize_finite_scalar(self.max_scale, "max_scale")
        ewma_alpha = _normalize_finite_scalar(self.ewma_alpha, "ewma_alpha")
        high_nis_ratio = _normalize_finite_scalar(
            self.high_nis_ratio,
            "high_nis_ratio",
        )
        low_nis_ratio = _normalize_finite_scalar(
            self.low_nis_ratio,
            "low_nis_ratio",
        )
        scale_gain = _normalize_finite_scalar(self.scale_gain, "scale_gain")

        if base_scale <= 0.0:
            raise ValueError("base_scale must be positive")
        if min_scale <= 0.0 or max_scale < min_scale:
            raise ValueError("scale bounds must be positive and ordered")
        if not 0.0 < ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1]")
        if high_nis_ratio < low_nis_ratio:
            raise ValueError("high_nis_ratio must be at least low_nis_ratio")
        if scale_gain < 0.0:
            raise ValueError("scale_gain must be nonnegative")

        object.__setattr__(self, "base_scale", base_scale)
        object.__setattr__(self, "min_scale", min_scale)
        object.__setattr__(self, "max_scale", max_scale)
        object.__setattr__(self, "ewma_alpha", ewma_alpha)
        object.__setattr__(self, "high_nis_ratio", high_nis_ratio)
        object.__setattr__(self, "low_nis_ratio", low_nis_ratio)
        object.__setattr__(self, "scale_gain", scale_gain)


def _normalize_finite_scalar(value: float, name: str) -> float:
    value_array = np.asarray(value)
    if value_array.shape != () or value_array.dtype == np.bool_:
        raise ValueError(f"{name} must be a finite scalar")
    value_scalar = value_array.item()
    if isinstance(value_scalar, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite scalar")
    try:
        value_float = float(value_scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(value_float):
        raise ValueError(f"{name} must be a finite scalar")
    return value_float


def _normalize_nonnegative_finite_scalar(value: float, name: str) -> float:
    message = f"{name} must be a nonnegative finite scalar"
    value_array = np.asarray(value)
    if (
        value_array.shape != ()
        or value_array.dtype == np.bool_
        or value_array.dtype.kind in "USbcMm"
    ):
        raise ValueError(message)
    value_scalar = value_array.item()
    if isinstance(value_scalar, (bool, np.bool_, str, bytes, bytearray)):
        raise ValueError(message)
    try:
        value_float = float(value_scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if value_float < 0.0 or not np.isfinite(value_float):
        raise ValueError(message)
    return value_float


@dataclass
class RollingNISProcessNoiseAdapter:
    """Maintain EWMA NIS ratios and return process-noise scale factors."""

    config: AdaptiveProcessNoiseConfig = field(
        default_factory=AdaptiveProcessNoiseConfig
    )
    ratios_by_source: dict[str, float] = field(default_factory=dict)
    updates_by_source: dict[str, int] = field(default_factory=dict)

    def observe(
        self,
        *,
        source: str = "default",
        measurement_dim: int,
        nis: float,
        accepted: bool = True,
    ) -> float:
        """Ingest one innovation and return the updated source NIS ratio."""

        source = str(source)
        measurement_dim = int(measurement_dim)
        nis = float(nis)
        if not accepted or measurement_dim <= 0 or not np.isfinite(nis):
            return self.ratios_by_source.get(source, 1.0)
        ratio = max(nis / float(measurement_dim), 0.0)
        previous = self.ratios_by_source.get(source, 1.0)
        alpha = float(self.config.ewma_alpha)
        updated = (1.0 - alpha) * previous + alpha * ratio
        self.ratios_by_source[source] = float(updated)
        self.updates_by_source[source] = self.updates_by_source.get(source, 0) + 1
        return float(updated)

    def ratio(self, source_weights: Mapping[str, float] | None = None) -> float:
        """Return a weighted aggregate NIS ratio across observed sources."""

        if not self.ratios_by_source:
            return 1.0
        if source_weights:
            numerator = 0.0
            denominator = 0.0
            for source, ratio in self.ratios_by_source.items():
                weight = float(source_weights.get(source, 0.0))
                if weight > 0.0:
                    numerator += weight * ratio
                    denominator += weight
            if denominator > 0.0:
                return float(numerator / denominator)
        return float(np.mean(list(self.ratios_by_source.values())))

    def scale(self, source_weights: Mapping[str, float] | None = None) -> float:
        """Return the adapted process-noise multiplier."""

        return float(
            self.config.base_scale
            * adaptive_scale_from_ratio(self.ratio(source_weights), self.config)
        )

    def scaled_covariance(
        self,
        process_noise_covariance: np.ndarray,
        source_weights: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """Return ``process_noise_covariance`` multiplied by the adapted scale."""

        return np.asarray(process_noise_covariance, dtype=float) * self.scale(
            source_weights
        )


def adaptive_scale_from_ratio(
    ratio: float, config: AdaptiveProcessNoiseConfig | None = None
) -> float:
    """Map a normalized NIS ratio to a bounded process-noise scale."""

    config = AdaptiveProcessNoiseConfig() if config is None else config
    ratio = _normalize_nonnegative_finite_scalar(ratio, "ratio")
    if ratio > float(config.high_nis_ratio):
        scale = 1.0 + float(config.scale_gain) * (ratio - float(config.high_nis_ratio))
    elif ratio < float(config.low_nis_ratio):
        scale = 1.0 - float(config.scale_gain) * (float(config.low_nis_ratio) - ratio)
    else:
        scale = 1.0
    return float(np.clip(scale, float(config.min_scale), float(config.max_scale)))


RollingNISAdaptiveProcessNoise = RollingNISProcessNoiseAdapter
adaptive_process_noise_scale_from_nis_ratio = adaptive_scale_from_ratio

__all__ = [
    "AdaptiveProcessNoiseConfig",
    "RollingNISAdaptiveProcessNoise",
    "RollingNISProcessNoiseAdapter",
    "adaptive_process_noise_scale_from_nis_ratio",
    "adaptive_scale_from_ratio",
]
