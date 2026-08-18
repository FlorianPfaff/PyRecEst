from types import SimpleNamespace

import numpy as np
import pytest

from pyrecest.filters.survival_aware_association import (
    SurvivalAwareAssociationConfig,
    survival_aware_track_log_prior,
)


@pytest.mark.parametrize(
    "invalid",
    [True, np.bool_(True), "0.8", np.array(0.8 + 0.1j), np.ma.masked],
)
def test_config_rejects_non_real_probability_scalars(invalid):
    with pytest.raises(
        ValueError, match="survival_probability must be a scalar probability"
    ):
        SurvivalAwareAssociationConfig(survival_probability=invalid)


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("appearance_likelihood", True, "appearance_likelihood must be a scalar likelihood"),
        ("mass_power", "1.0", "mass_power must be a scalar number"),
        ("birth_weight", np.array(1.0 + 0.25j), "birth_weight must be a scalar number"),
    ],
)
def test_config_rejects_non_real_nonnegative_scalars(field, invalid, message):
    with pytest.raises(ValueError, match=message):
        SurvivalAwareAssociationConfig(**{field: invalid})


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("survival_probability", True, "survival_probability must be a scalar probability"),
        (
            "appearance_likelihood",
            np.array(0.8 + 0.1j),
            "appearance_likelihood must be a scalar likelihood",
        ),
    ],
)
def test_callable_factors_preserve_type_until_validation(field, invalid, message):
    config = SurvivalAwareAssociationConfig(**{field: lambda track: invalid})
    track = SimpleNamespace(hits=1, misses=0, metadata={})

    with pytest.raises(ValueError, match=message):
        survival_aware_track_log_prior(track, config=config)
