"""Compatibility import for the canonical rolling NIS adapter name.

The implementation lives in :mod:`pyrecest.filters.adaptive_process_noise`.
This module keeps the class discoverable through the conventional module path
that matches its public class name.
"""

from .adaptive_process_noise import (
    AdaptiveProcessNoiseConfig,
    RollingNISAdaptiveProcessNoise,
    RollingNISProcessNoiseAdapter,
    adaptive_process_noise_scale_from_nis_ratio,
    adaptive_scale_from_ratio,
)

__all__ = [
    "AdaptiveProcessNoiseConfig",
    "RollingNISAdaptiveProcessNoise",
    "RollingNISProcessNoiseAdapter",
    "adaptive_process_noise_scale_from_nis_ratio",
    "adaptive_scale_from_ratio",
]
