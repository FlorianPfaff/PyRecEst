from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pytest

from pyrecest.models import (
    MaskedLinearMeasurementModel,
    WeakDimensionMeasurementModel,
    block_diag_measurement_covariance,
    diagonal_measurement_covariance,
)


_NON_REAL_STD_CASES = (
    [1.0 + 0.0j],
    [np.complex128(1.0 + 0.0j)],
    [np.datetime64("2024-01-01")],
    [np.timedelta64(1, "D")],
    [date(2024, 1, 1)],
    [datetime(2024, 1, 1, 12, 0, 0)],
    [timedelta(days=1)],
)


@pytest.mark.parametrize("stds", _NON_REAL_STD_CASES)
def test_measurement_stds_reject_non_real_values(stds) -> None:
    with pytest.raises(ValueError, match="real numeric values"):
        diagonal_measurement_covariance(stds)
    with pytest.raises(ValueError, match="real numeric values"):
        block_diag_measurement_covariance(trusted_std=stds)
    with pytest.raises(ValueError, match="real numeric values"):
        MaskedLinearMeasurementModel(state_dim=1, observed_dims=[0], stds=stds)
    with pytest.raises(ValueError, match="real numeric values"):
        WeakDimensionMeasurementModel(np.eye(1), stds=stds)


@pytest.mark.parametrize("std_value", tuple(case[0] for case in _NON_REAL_STD_CASES))
def test_measurement_std_mappings_reject_non_real_values(std_value) -> None:
    with pytest.raises(ValueError, match="real numeric values"):
        block_diag_measurement_covariance(trusted_std={"x": std_value})
    with pytest.raises(ValueError, match="real numeric values"):
        WeakDimensionMeasurementModel(np.eye(1), stds={"x": std_value})


@pytest.mark.parametrize(
    "measurement_noise_cov",
    (
        np.array([[1.0 + 0.0j]]),
        np.array([[np.complex128(1.0 + 0.0j)]], dtype=object),
        np.array([[np.datetime64("2024-01-01")]]),
        np.array([[np.timedelta64(1, "D")]]),
        np.array([[date(2024, 1, 1)]], dtype=object),
        np.array([[datetime(2024, 1, 1, 12, 0, 0)]], dtype=object),
        np.array([[timedelta(days=1)]], dtype=object),
    ),
)
def test_measurement_noise_cov_rejects_non_real_values(measurement_noise_cov) -> None:
    with pytest.raises(ValueError, match="measurement_noise_cov"):
        MaskedLinearMeasurementModel(
            state_dim=1,
            observed_dims=[0],
            measurement_noise_cov=measurement_noise_cov,
        )
    with pytest.raises(ValueError, match="measurement_noise_cov"):
        WeakDimensionMeasurementModel(
            np.eye(1),
            measurement_noise_cov=measurement_noise_cov,
        )
