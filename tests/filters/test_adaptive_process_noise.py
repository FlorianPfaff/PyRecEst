import numpy as np

from pyrecest.filters.adaptive_process_noise import (
    AdaptiveProcessNoiseConfig,
    RollingNISProcessNoiseAdapter,
    adaptive_scale_from_ratio,
)


def test_scale_increases_for_high_nis_ratio_and_decreases_for_low_ratio():
    config = AdaptiveProcessNoiseConfig(min_scale=0.5, max_scale=3.0, high_nis_ratio=1.5, low_nis_ratio=0.5, scale_gain=1.0)
    assert adaptive_scale_from_ratio(2.0, config) > 1.0
    assert adaptive_scale_from_ratio(0.0, config) < 1.0


def test_rolling_adapter_updates_source_ratio_and_scales_covariance():
    adapter = RollingNISProcessNoiseAdapter(AdaptiveProcessNoiseConfig(ewma_alpha=1.0, high_nis_ratio=1.1))
    ratio = adapter.observe(source="radar", measurement_dim=2, nis=6.0)
    assert np.isclose(ratio, 3.0)
    scaled = adapter.scaled_covariance(np.eye(2), {"radar": 1.0})
    assert np.allclose(scaled, np.eye(2) * adapter.scale({"radar": 1.0}))
