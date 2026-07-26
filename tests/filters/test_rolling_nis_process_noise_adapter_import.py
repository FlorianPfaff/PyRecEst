"""Regression tests for the canonical adaptive-process-noise import path."""


def test_canonical_rolling_nis_adapter_module_reexports_implementation():
    from pyrecest.filters.adaptive_process_noise import (
        AdaptiveProcessNoiseConfig as ImplementationConfig,
    )
    from pyrecest.filters.adaptive_process_noise import (
        RollingNISProcessNoiseAdapter as ImplementationAdapter,
    )
    from pyrecest.filters.adaptive_process_noise import (
        adaptive_scale_from_ratio as implementation_scale,
    )
    from pyrecest.filters.rolling_nis_process_noise_adapter import (
        AdaptiveProcessNoiseConfig,
        RollingNISProcessNoiseAdapter,
        adaptive_scale_from_ratio,
    )

    assert RollingNISProcessNoiseAdapter is ImplementationAdapter
    assert AdaptiveProcessNoiseConfig is ImplementationConfig
    assert adaptive_scale_from_ratio is implementation_scale
