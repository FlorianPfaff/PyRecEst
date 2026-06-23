from __future__ import annotations

import numpy as np
import pytest
from pyrecest.filters import KalmanFilter
from pyrecest.models import (
    MaskedLinearMeasurementModel,
    WeakDimensionMeasurementModel,
    block_diag_measurement_covariance,
    diagonal_measurement_covariance,
    selection_matrix,
)


def test_diagonal_measurement_covariance_from_stds() -> None:
    covariance = diagonal_measurement_covariance([2.0, 3.0])

    assert np.allclose(covariance, np.diag([4.0, 9.0]))


def test_block_diag_measurement_covariance_preserves_named_order() -> None:
    covariance = block_diag_measurement_covariance(
        trusted_std={"x": 1.0, "y": 2.0},
        weak_std={"z": 100.0},
        dimension_order=["x", "y", "z"],
    )

    assert np.allclose(covariance, np.diag([1.0, 4.0, 10000.0]))


def test_block_diag_measurement_covariance_rejects_duplicate_dimension_order() -> None:
    with pytest.raises(ValueError, match="dimension_order"):
        block_diag_measurement_covariance(
            trusted_std={"x": 1.0, "y": 2.0},
            dimension_order=["x", "x"],
        )


def test_selection_matrix_selects_state_components() -> None:
    matrix = selection_matrix(6, [0, 2, 5])

    assert matrix.shape == (3, 6)
    assert np.allclose(
        matrix @ np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), [1.0, 3.0, 6.0]
    )


def test_selection_matrix_rejects_bool_and_vector_indices() -> None:
    invalid_cases = (
        (True, [0], "state_dim"),
        (3, [False], "observed_dims"),
        (3, [np.array([1])], "observed_dims"),
    )

    for state_dim, observed_dims, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            selection_matrix(state_dim, observed_dims)


def test_masked_linear_measurement_model_updates_only_observed_dimensions() -> None:
    model = MaskedLinearMeasurementModel(
        state_dim=3, observed_dims=[0, 2], stds=[1.0, 2.0]
    )
    kf = KalmanFilter((np.zeros(3), np.eye(3) * 100.0))

    kf.update_model(model, np.array([10.0, 20.0]))
    estimate = np.asarray(kf.get_point_estimate(), dtype=float)

    assert estimate[0] > 9.0
    assert estimate[1] == 0.0
    assert estimate[2] > 15.0


def test_masked_linear_measurement_model_preserves_generator_observed_dims() -> None:
    model = MaskedLinearMeasurementModel(
        state_dim=3,
        observed_dims=(dim for dim in [0, 2]),
        stds=[1.0, 2.0],
    )

    assert model.observed_dims == (0, 2)
    assert np.allclose(model.matrix, selection_matrix(3, [0, 2]))


def test_weak_dimension_measurement_model_keeps_weak_dimension_nearly_untrusted() -> (
    None
):
    model = WeakDimensionMeasurementModel(np.eye(3), stds=[1.0, 1.0, 20000.0])
    kf = KalmanFilter((np.zeros(3), np.eye(3)))

    kf.update_model(model, np.array([10.0, 10.0, 1000.0]))
    estimate = np.asarray(kf.get_point_estimate(), dtype=float)

    assert estimate[0] > 4.0
    assert estimate[1] > 4.0
    assert abs(estimate[2]) < 1.0e-3
