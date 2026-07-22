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
