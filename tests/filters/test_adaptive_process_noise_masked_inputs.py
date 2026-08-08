import numpy as np
import pytest

from pyrecest.filters.adaptive_process_noise import (
    AdaptiveProcessNoiseConfig,
    RollingNISProcessNoiseAdapter,
    adaptive_scale_from_ratio,
)


def test_config_and_ratio_reject_masked_scalars():
    with pytest.raises(ValueError, match="base_scale must be a finite scalar"):
        AdaptiveProcessNoiseConfig(base_scale=np.ma.array(1.25, mask=True))

    with pytest.raises(
        ValueError, match="ratio must be a nonnegative finite scalar"
    ):
        adaptive_scale_from_ratio(np.ma.array(2.0, mask=True))


def test_observe_rejects_masked_controls_before_mutation():
    adapter = RollingNISProcessNoiseAdapter()

    for kwargs, message in (
        (
            {"measurement_dim": np.ma.array(2, mask=True), "nis": 4.0},
            "measurement_dim must be a positive integer",
        ),
        (
            {"measurement_dim": 2, "nis": np.ma.array(4.0, mask=True)},
            "nis must be a nonnegative finite scalar",
        ),
        (
            {
                "measurement_dim": 2,
                "nis": 4.0,
                "accepted": np.ma.array(True, mask=True),
            },
            "accepted must be a boolean",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            adapter.observe(**kwargs)
        assert adapter.ratios_by_source == {}
        assert adapter.updates_by_source == {}


def test_source_weight_rejects_masked_scalar():
    adapter = RollingNISProcessNoiseAdapter()
    adapter.observe(source="radar", measurement_dim=2, nis=4.0)

    with pytest.raises(
        ValueError,
        match=r"source_weights\['radar'\] must be a nonnegative finite scalar",
    ):
        adapter.ratio({"radar": np.ma.array(1.0, mask=True)})


@pytest.mark.parametrize(
    "covariance",
    [
        np.ma.array(
            [[1.0, 0.0], [0.0, 1.0]],
            mask=[[False, True], [False, False]],
        ),
        [[1.0, np.ma.masked], [0.0, 1.0]],
        np.array([[1.0, np.ma.masked], [0.0, 1.0]], dtype=object),
    ],
)
def test_scaled_covariance_rejects_masked_values(covariance):
    adapter = RollingNISProcessNoiseAdapter()

    with pytest.raises(
        ValueError,
        match="process_noise_covariance must contain only finite real numeric values",
    ):
        adapter.scaled_covariance(covariance)


def test_clear_mask_wrappers_remain_supported():
    config = AdaptiveProcessNoiseConfig(
        base_scale=np.ma.array(1.25, mask=False),
        ewma_alpha=np.ma.array(1.0, mask=False),
    )
    adapter = RollingNISProcessNoiseAdapter(config)

    ratio = adapter.observe(
        source="radar",
        measurement_dim=np.ma.array(2, mask=False),
        nis=np.ma.array(4.0, mask=False),
        accepted=np.ma.array(True, mask=False),
    )
    scaled = adapter.scaled_covariance(
        np.ma.array(np.eye(2), mask=False),
        {"radar": np.ma.array(1.0, mask=False)},
    )

    assert ratio == 2.0
    assert np.allclose(scaled, np.eye(2) * adapter.scale({"radar": 1.0}))
