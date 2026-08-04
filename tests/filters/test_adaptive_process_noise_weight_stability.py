import numpy as np
import pytest
from pyrecest.filters.adaptive_process_noise import (
    AdaptiveProcessNoiseConfig,
    RollingNISProcessNoiseAdapter,
)


def test_weighted_ratio_handles_extreme_finite_source_weights():
    adapter = RollingNISProcessNoiseAdapter(AdaptiveProcessNoiseConfig(ewma_alpha=1.0))
    adapter.observe(source="radar", measurement_dim=2, nis=6.0)
    adapter.observe(source="camera", measurement_dim=2, nis=2.0)

    max_float = np.finfo(float).max
    source_weights = {
        "radar": max_float,
        "camera": max_float / 2.0,
    }

    assert adapter.ratio(source_weights) == pytest.approx(7.0 / 3.0)
    assert np.isfinite(adapter.scale(source_weights))


def test_unweighted_ratio_handles_extreme_finite_source_ratios():
    adapter = RollingNISProcessNoiseAdapter(AdaptiveProcessNoiseConfig(ewma_alpha=1.0))
    max_float = np.finfo(float).max
    adapter.observe(source="radar", measurement_dim=1, nis=max_float)
    adapter.observe(source="camera", measurement_dim=2, nis=max_float)

    expected_ratio = 0.75 * max_float

    assert adapter.ratio() == pytest.approx(expected_ratio)
    assert adapter.scale() == pytest.approx(adapter.config.max_scale)
