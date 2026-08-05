from __future__ import annotations

import numpy as np
import pytest

from pyrecest.smoothers.robust_linear_gaussian_map import (
    LinearGaussianMeasurementFactor,
    RobustLinearGaussianMapConfig,
    fixed_lag_robust_linear_gaussian_map_smooth,
    robust_linear_gaussian_map_smooth,
)


def test_linear_loss_matches_dense_weighted_least_squares() -> None:
    initial = np.zeros((3, 1))
    transitions = np.ones((2, 1, 1))
    process_covariances = np.full((2, 1, 1), 0.1)
    factors = tuple(
        LinearGaussianMeasurementFactor(
            state_index=index,
            measurement=np.array([value]),
            observation_matrix=np.ones((1, 1)),
            covariance=np.array([[0.2]]),
        )
        for index, value in enumerate((0.0, 1.0, 2.0))
    )

    result = robust_linear_gaussian_map_smooth(
        initial,
        prior_mean=np.array([0.0]),
        prior_covariance=np.array([[1.0]]),
        transition_matrices=transitions,
        process_covariances=process_covariances,
        measurements=factors,
        config=RobustLinearGaussianMapConfig(loss="linear"),
    )

    prior_weight = 1.0
    process_weight = 1.0 / np.sqrt(0.1)
    measurement_weight = 1.0 / np.sqrt(0.2)
    matrix = np.array(
        [
            [prior_weight, 0.0, 0.0],
            [-process_weight, process_weight, 0.0],
            [0.0, -process_weight, process_weight],
            [measurement_weight, 0.0, 0.0],
            [0.0, measurement_weight, 0.0],
            [0.0, 0.0, measurement_weight],
        ]
    )
    rhs = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            measurement_weight,
            2.0 * measurement_weight,
        ]
    )
    expected = np.linalg.lstsq(matrix, rhs, rcond=None)[0]

    assert result.success
    assert result.covariances is None
    assert np.allclose(result.states[:, 0], expected, atol=1.0e-7)
    assert result.final_cost <= result.initial_cost


def test_huber_loss_reduces_single_measurement_outlier_error() -> None:
    times = np.arange(7, dtype=float)
    truth = np.column_stack((times, np.ones_like(times)))
    measurements = times.copy()
    measurements[3] += 50.0
    initial = np.column_stack((measurements, np.ones_like(times)))
    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    transitions = np.repeat(transition[None, :, :], len(times) - 1, axis=0)
    process_covariances = np.repeat(
        np.diag([0.05, 0.05])[None, :, :],
        len(times) - 1,
        axis=0,
    )
    factors = tuple(
        LinearGaussianMeasurementFactor(
            state_index=index,
            measurement=np.array([measurement]),
            observation_matrix=np.array([[1.0, 0.0]]),
            covariance=np.array([[1.0]]),
        )
        for index, measurement in enumerate(measurements)
    )

    linear = robust_linear_gaussian_map_smooth(
        initial,
        prior_mean=np.array([0.0, 1.0]),
        prior_covariance=np.diag([0.1, 0.1]),
        transition_matrices=transitions,
        process_covariances=process_covariances,
        measurements=factors,
        config=RobustLinearGaussianMapConfig(loss="linear"),
    )
    robust = robust_linear_gaussian_map_smooth(
        initial,
        prior_mean=np.array([0.0, 1.0]),
        prior_covariance=np.diag([0.1, 0.1]),
        transition_matrices=transitions,
        process_covariances=process_covariances,
        measurements=factors,
        config=RobustLinearGaussianMapConfig(loss="huber", loss_scale=2.0),
    )

    linear_rmse = np.sqrt(np.mean((linear.states[:, 0] - truth[:, 0]) ** 2))
    robust_rmse = np.sqrt(np.mean((robust.states[:, 0] - truth[:, 0]) ** 2))
    assert robust.success
    assert robust_rmse < 0.25 * linear_rmse
    assert robust.measurement_sqrt_weights[3] < 0.5
    assert robust.final_cost <= robust.initial_cost


def test_measurement_offsets_and_vector_measurements_are_supported() -> None:
    factor = LinearGaussianMeasurementFactor(
        state_index=0,
        measurement=np.array([3.0, 6.0]),
        observation_matrix=np.eye(2),
        covariance=np.eye(2) * 0.01,
        offset=np.array([1.0, 2.0]),
        robust=False,
    )
    result = robust_linear_gaussian_map_smooth(
        np.zeros((1, 2)),
        prior_mean=np.zeros(2),
        prior_covariance=np.eye(2) * 100.0,
        transition_matrices=np.empty((0, 2, 2)),
        process_covariances=np.empty((0, 2, 2)),
        measurements=(factor,),
    )

    assert np.allclose(result.states[0], np.array([2.0, 4.0]), atol=1.0e-3)


def test_fixed_lag_returns_one_window_summary_per_state() -> None:
    times = np.arange(4, dtype=float)
    initial = np.arange(4, dtype=float).reshape(-1, 1)
    transitions = np.ones((3, 1, 1))
    process_covariances = np.full((3, 1, 1), 0.2)
    factors = tuple(
        LinearGaussianMeasurementFactor(
            state_index=index,
            measurement=np.array([value]),
            observation_matrix=np.ones((1, 1)),
            covariance=np.array([[0.1]]),
        )
        for index, value in enumerate((0.0, 1.2, 2.1, 3.0))
    )

    result = fixed_lag_robust_linear_gaussian_map_smooth(
        times,
        initial,
        anchor_covariances=np.repeat(np.array([[[1.0]]]), 4, axis=0),
        transition_matrices=transitions,
        process_covariances=process_covariances,
        measurements=factors,
        lag=1.5,
        config=RobustLinearGaussianMapConfig(loss="linear"),
    )

    assert result.states.shape == initial.shape
    assert len(result.windows) == len(times)
    assert result.windows[0].start_index == 0
    assert result.windows[0].end_index == 1
    assert result.windows[-1].message == "window contains no future state"
    assert result.states[-1, 0] == initial[-1, 0]
    assert not np.allclose(result.states[1, 0], initial[1, 0])


def test_fixed_lag_covering_all_future_matches_batch_first_state() -> None:
    times = np.arange(3, dtype=float)
    initial = np.array([[0.0], [1.3], [2.0]])
    transitions = np.ones((2, 1, 1))
    process_covariances = np.full((2, 1, 1), 0.1)
    factors = tuple(
        LinearGaussianMeasurementFactor(
            state_index=index,
            measurement=np.array([value]),
            observation_matrix=np.ones((1, 1)),
            covariance=np.array([[0.2]]),
        )
        for index, value in enumerate((0.0, 1.0, 2.0))
    )
    batch = robust_linear_gaussian_map_smooth(
        initial,
        prior_mean=initial[0],
        prior_covariance=np.array([[1.0]]),
        transition_matrices=transitions,
        process_covariances=process_covariances,
        measurements=factors,
        config=RobustLinearGaussianMapConfig(loss="linear"),
    )
    lagged = fixed_lag_robust_linear_gaussian_map_smooth(
        times,
        initial,
        anchor_covariances=np.repeat(np.array([[[1.0]]]), 3, axis=0),
        transition_matrices=transitions,
        process_covariances=process_covariances,
        measurements=factors,
        lag=100.0,
        config=RobustLinearGaussianMapConfig(loss="linear"),
    )

    assert np.allclose(lagged.states[0], batch.states[0])


@pytest.mark.parametrize("bad_lag", [-1.0, np.nan, np.inf, True, np.array([1.0])])
def test_fixed_lag_rejects_invalid_lag(bad_lag) -> None:
    with pytest.raises(ValueError):
        fixed_lag_robust_linear_gaussian_map_smooth(
            np.array([0.0, 1.0]),
            np.zeros((2, 1)),
            anchor_covariances=np.repeat(np.array([[[1.0]]]), 2, axis=0),
            transition_matrices=np.ones((1, 1, 1)),
            process_covariances=np.ones((1, 1, 1)),
            lag=bad_lag,
        )


def test_single_state_accepts_empty_transition_sequences() -> None:
    result = robust_linear_gaussian_map_smooth(
        np.array([[2.0]]),
        prior_mean=np.array([1.0]),
        prior_covariance=np.array([[1.0]]),
        transition_matrices=(),
        process_covariances=(),
        config=RobustLinearGaussianMapConfig(loss="linear"),
    )

    assert result.success
    assert result.message == "solved"
    assert np.allclose(result.states, np.array([[1.0]]))


def test_measurement_factor_must_match_state_dimension_and_index() -> None:
    bad_shape = LinearGaussianMeasurementFactor(
        state_index=0,
        measurement=np.array([1.0]),
        observation_matrix=np.ones((1, 2)),
        covariance=np.eye(1),
    )
    with pytest.raises(ValueError, match="column count"):
        robust_linear_gaussian_map_smooth(
            np.zeros((1, 1)),
            prior_mean=np.zeros(1),
            prior_covariance=np.eye(1),
            transition_matrices=np.empty((0, 1, 1)),
            process_covariances=np.empty((0, 1, 1)),
            measurements=(bad_shape,),
        )

    bad_index = LinearGaussianMeasurementFactor(
        state_index=1,
        measurement=np.array([1.0]),
        observation_matrix=np.ones((1, 1)),
        covariance=np.eye(1),
    )
    with pytest.raises(ValueError, match="outside"):
        robust_linear_gaussian_map_smooth(
            np.zeros((1, 1)),
            prior_mean=np.zeros(1),
            prior_covariance=np.eye(1),
            transition_matrices=np.empty((0, 1, 1)),
            process_covariances=np.empty((0, 1, 1)),
            measurements=(bad_index,),
        )
