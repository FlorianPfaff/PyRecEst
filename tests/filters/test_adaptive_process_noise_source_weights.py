"""Regression tests for weighted rolling NIS aggregation."""

import numpy as np
import pytest

from pyrecest.filters.adaptive_process_noise import (
    AdaptiveProcessNoiseConfig,
    RollingNISProcessNoiseAdapter,
)


def _adapter_with_two_sources():
    adapter = RollingNISProcessNoiseAdapter(
        AdaptiveProcessNoiseConfig(ewma_alpha=1.0)
    )
    adapter.observe(source="radar", measurement_dim=1, nis=2.0)
    adapter.observe(source="camera", measurement_dim=1, nis=4.0)
    return adapter


def test_weighted_ratio_remains_finite_for_maximum_finite_weights():
    adapter = _adapter_with_two_sources()
    weight = np.finfo(float).max

    ratio = adapter.ratio({"radar": weight, "camera": weight})

    assert np.isfinite(ratio)
    assert ratio == pytest.approx(3.0)


@pytest.mark.parametrize(
    "weight",
    [
        np.nan,
        np.inf,
        -np.inf,
        -1.0,
        True,
        "1.0",
        np.array([1.0]),
        np.timedelta64(1, "ns"),
    ],
)
def test_weighted_ratio_rejects_invalid_source_weights(weight):
    adapter = _adapter_with_two_sources()

    with pytest.raises(
        ValueError,
        match=r"source_weights\['radar'\] must be a nonnegative finite scalar",
    ):
        adapter.ratio({"radar": weight, "camera": 1.0})
