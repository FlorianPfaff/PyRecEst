import pytest
from pyrecest.filters.adaptive_process_noise import (
    AdaptiveProcessNoiseConfig,
    RollingNISProcessNoiseAdapter,
    adaptive_scale_from_ratio,
)


@pytest.mark.parametrize(
    ("base_scale", "min_scale", "max_scale"),
    (
        (0.5, 0.25, 0.8),
        (2.0, 1.5, 3.0),
    ),
)
def test_nominal_ratio_preserves_base_scale_when_bounds_exclude_one(
    base_scale,
    min_scale,
    max_scale,
):
    config = AdaptiveProcessNoiseConfig(
        base_scale=base_scale,
        min_scale=min_scale,
        max_scale=max_scale,
    )

    assert adaptive_scale_from_ratio(1.0, config) == pytest.approx(base_scale)
    assert RollingNISProcessNoiseAdapter(config).scale() == pytest.approx(base_scale)


def test_relative_adaptation_is_applied_before_absolute_bounds():
    config = AdaptiveProcessNoiseConfig(
        base_scale=0.5,
        min_scale=0.25,
        max_scale=0.8,
        high_nis_ratio=1.5,
        scale_gain=0.5,
    )

    assert adaptive_scale_from_ratio(2.0, config) == pytest.approx(0.625)


def test_absolute_lower_bound_is_applied_after_relative_adaptation():
    config = AdaptiveProcessNoiseConfig(
        base_scale=2.0,
        min_scale=1.5,
        max_scale=3.0,
        low_nis_ratio=0.6,
        scale_gain=0.5,
    )

    assert adaptive_scale_from_ratio(0.0, config) == pytest.approx(1.5)
