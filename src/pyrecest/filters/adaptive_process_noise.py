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
        if self.base_scale <= 0.0:
            raise ValueError("base_scale must be positive")
        if self.min_scale <= 0.0 or self.max_scale < self.min_scale:
            raise ValueError("scale bounds must be positive and ordered")
        if not 0.0 < self.ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1]")
        if self.high_nis_ratio < self.low_nis_ratio:
            raise ValueError("high_nis_ratio must be at least low_nis_ratio")
        if self.scale_gain < 0.0:
            raise ValueError("scale_gain must be nonnegative")


@dataclass
class RollingNISProcessNoiseAdapter:
    """Maintain EWMA NIS ratios and return process-noise scale factors."""

    config: AdaptiveProcessNoiseConfig = field(default_factory=AdaptiveProcessNoiseConfig)
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

        return float(self.config.base_scale * adaptive_scale_from_ratio(self.ratio(source_weights), self.config))

    def scaled_covariance(
        self,
        process_noise_covariance: np.ndarray,
        source_weights: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """Return ``process_noise_covariance`` multiplied by the adapted scale."""

        return np.asarray(process_noise_covariance, dtype=float) * self.scale(source_weights)


def adaptive_scale_from_ratio(ratio: float, config: AdaptiveProcessNoiseConfig | None = None) -> float:
    """Map a normalized NIS ratio to a bounded process-noise scale."""

    config = AdaptiveProcessNoiseConfig() if config is None else config
    ratio = float(ratio)
    if ratio > float(config.high_nis_ratio):
        scale = 1.0 + float(config.scale_gain) * (ratio - float(config.high_nis_ratio))
    elif ratio < float(config.low_nis_ratio):
        scale = 1.0 - float(config.scale_gain) * (float(config.low_nis_ratio) - ratio)
    else:
        scale = 1.0
    return float(np.clip(scale, float(config.min_scale), float(config.max_scale)))


__all__ = [
    "AdaptiveProcessNoiseConfig",
    "RollingNISProcessNoiseAdapter",
    "adaptive_scale_from_ratio",
]
