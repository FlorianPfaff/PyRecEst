import numpy as np
import pytest
from pyrecest.filters.adaptive_process_noise import (
    AdaptiveProcessNoiseConfig,
    RollingNISProcessNoiseAdapter,
    adaptive_scale_from_ratio,
)


def test_scale_increases_for_high_nis_ratio_and_decreases_for_low_ratio():
    config = AdaptiveProcessNoiseConfig(
        min_scale=0.5,
        max_scale=3.0,
        high_nis_ratio=1.5,
        low_nis_ratio=0.5,
        scale_gain=1.0,
    )
    assert adaptive_scale_from_ratio(2.0, config) > 1.0
    assert adaptive_scale_from_ratio(0.0, config) < 1.0


@pytest.mark.parametrize(
    "ratio",
    [
        np.nan,
        np.inf,
        -np.inf,
        -0.1,
        True,
        "1.0",
        np.array([1.0]),
        np.array(np.timedelta64(2, "ns"), dtype=object),
        np.array(np.datetime64("1970-01-01T00:00:00.000000001"), dtype=object),
    ],
)
def test_scale_rejects_invalid_ratios(ratio):
    with pytest.raises(ValueError, match="ratio must be a nonnegative finite scalar"):
        adaptive_scale_from_ratio(ratio)


def test_rolling_adapter_updates_source_ratio_and_scales_covariance():
    adapter = RollingNISProcessNoiseAdapter(
        AdaptiveProcessNoiseConfig(ewma_alpha=1.0, high_nis_ratio=1.1)
    )
    ratio = adapter.observe(source="radar", measurement_dim=2, nis=6.0)
    assert np.isclose(ratio, 3.0)
    scaled = adapter.scaled_covariance(np.eye(2), {"radar": 1.0})
    assert np.allclose(scaled, np.eye(2) * adapter.scale({"radar": 1.0}))


@pytest.mark.parametrize(
    "covariance",
    [
        np.array([[1.0 + 2.0j]]),
        np.array([[True]]),
        np.array([[np.nan]]),
        np.array([[np.inf]]),
        np.array([[1.0 + 2.0j]], dtype=object),
        np.array([[np.timedelta64(1, "ns")]], dtype=object),
    ],
)
def test_scaled_covariance_rejects_nonreal_or_nonfinite_values(covariance):
    adapter = RollingNISProcessNoiseAdapter()

    with pytest.raises(
        ValueError,
        match="process_noise_covariance must contain only finite real numeric values",
    ):
        adapter.scaled_covariance(covariance)


def test_config_normalizes_scalar_numeric_values():
    config = AdaptiveProcessNoiseConfig(
        base_scale=np.array(1.25),
        min_scale=np.array(0.5),
        max_scale=np.array(2.5),
        ewma_alpha=np.array(0.25),
        high_nis_ratio=np.array(1.5),
        low_nis_ratio=np.array(0.5),
        scale_gain=np.array(0.75),
    )

    assert config.base_scale == 1.25
    assert config.min_scale == 0.5
    assert config.max_scale == 2.5
    assert config.ewma_alpha == 0.25
    assert config.high_nis_ratio == 1.5
    assert config.low_nis_ratio == 0.5
    assert config.scale_gain == 0.75


def test_config_rejects_nonfinite_or_nonscalar_values():
    invalid_values = (
        np.nan,
        np.inf,
        -np.inf,
        True,
        "1.0",
        b"1.0",
        np.array([1.0]),
        np.timedelta64(2, "ns"),
        np.array(np.timedelta64(2, "ns"), dtype=object),
    )

    for field_name in (
        "base_scale",
        "min_scale",
        "max_scale",
        "ewma_alpha",
        "high_nis_ratio",
        "low_nis_ratio",
        "scale_gain",
    ):
        for value in invalid_values:
            with pytest.raises(
                ValueError, match=f"{field_name} must be a finite scalar"
            ):
                AdaptiveProcessNoiseConfig(**{field_name: value})


def test_config_rejects_non_numeric_scalar_payloads():
    invalid_values = (
        "1.0",
        b"1.0",
        np.str_("1.0"),
        np.bytes_(b"1.0"),
        np.timedelta64(1, "ns"),
        np.datetime64("1970-01-01T00:00:00.000000001"),
        np.array(np.timedelta64(1, "ns"), dtype=object),
        np.array(np.datetime64("1970-01-01T00:00:00.000000001"), dtype=object),
    )

    for field_name in (
        "base_scale",
        "min_scale",
        "max_scale",
        "ewma_alpha",
        "high_nis_ratio",
        "low_nis_ratio",
        "scale_gain",
    ):
        for value in invalid_values:
            with pytest.raises(
                ValueError, match=f"{field_name} must be a finite scalar"
            ):
                AdaptiveProcessNoiseConfig(**{field_name: value})


def test_observe_rejects_temporal_scalar_controls():
    adapter = RollingNISProcessNoiseAdapter()

    for value in (
        np.timedelta64(2, "ns"),
        np.datetime64("1970-01-01T00:00:00.000000002"),
    ):
        with pytest.raises(
            ValueError, match="measurement_dim must be a positive integer"
        ):
            adapter.observe(measurement_dim=value, nis=1.0)

    for value in (
        np.array(np.timedelta64(1, "ns"), dtype=object),
        np.array(np.datetime64("1970-01-01T00:00:00.000000001"), dtype=object),
    ):
        with pytest.raises(ValueError, match="nis must be a nonnegative finite scalar"):
            adapter.observe(measurement_dim=2, nis=value)
