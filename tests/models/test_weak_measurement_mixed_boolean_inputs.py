from __future__ import annotations

import numpy as np
import pytest
from pyrecest.models import (
    MaskedLinearMeasurementModel,
    WeakDimensionMeasurementModel,
    block_diag_measurement_covariance,
    diagonal_measurement_covariance,
)


@pytest.mark.parametrize(
    "stds",
    (
        [1.0, True],
        (2.0, np.bool_(False)),
        [[1.0, True]],
    ),
)
def test_mixed_boolean_standard_deviations_are_rejected(stds) -> None:
    with pytest.raises(ValueError, match="real numeric values"):
        diagonal_measurement_covariance(stds)
    with pytest.raises(ValueError, match="real numeric values"):
        block_diag_measurement_covariance(trusted_std=stds)
    with pytest.raises(ValueError, match="real numeric values"):
        MaskedLinearMeasurementModel(
            state_dim=2,
            observed_dims=[0, 1],
            stds=stds,
        )
    with pytest.raises(ValueError, match="real numeric values"):
        WeakDimensionMeasurementModel(np.eye(2), stds=stds)


@pytest.mark.parametrize(
    "measurement_noise_cov",
    (
        [[1.0, 0.0], [True, 2.0]],
        ((1.0, np.bool_(False)), (0.0, 2.0)),
    ),
)
def test_mixed_boolean_measurement_covariances_are_rejected(
    measurement_noise_cov,
) -> None:
    with pytest.raises(ValueError, match="measurement_noise_cov"):
        MaskedLinearMeasurementModel(
            state_dim=2,
            observed_dims=[0, 1],
            measurement_noise_cov=measurement_noise_cov,
        )
    with pytest.raises(ValueError, match="measurement_noise_cov"):
        WeakDimensionMeasurementModel(
            np.eye(2),
            measurement_noise_cov=measurement_noise_cov,
        )


def test_mixed_python_numeric_sequences_remain_supported() -> None:
    covariance = diagonal_measurement_covariance([1, 2.5])
    assert np.allclose(covariance, np.diag([1.0, 6.25]))

    model = WeakDimensionMeasurementModel(
        [[1, 0.0], [0, 1.0]],
        measurement_noise_cov=[[1, 0.0], [0, 2.0]],
    )
    assert np.allclose(model.measurement_noise_cov, np.diag([1.0, 2.0]))
