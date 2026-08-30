import numpy as np
import pytest

from pyrecest import backend
from pyrecest.filters import KalmanFilter


def _one_dimensional_filter():
    return KalmanFilter((backend.array([0.0]), backend.array([[1.0]])))


def _assert_unit_state_unchanged(kalman_filter):
    np.testing.assert_allclose(backend.to_numpy(kalman_filter.filter_state.mu), [0.0])
    np.testing.assert_allclose(
        backend.to_numpy(kalman_filter.filter_state.C),
        [[1.0]],
    )


def test_predict_rejects_non_psd_process_noise_atomically():
    kalman_filter = _one_dimensional_filter()

    with pytest.raises(ValueError, match="sys_noise_cov must be positive semidefinite"):
        kalman_filter.predict_identity(backend.array([[-2.0]]))

    _assert_unit_state_unchanged(kalman_filter)


def test_update_rejects_non_psd_measurement_noise_atomically():
    kalman_filter = _one_dimensional_filter()

    with pytest.raises(ValueError, match="meas_noise must be positive semidefinite"):
        kalman_filter.update_identity(
            backend.array([[-0.5]]),
            backend.array([0.0]),
        )

    _assert_unit_state_unchanged(kalman_filter)


def test_innovation_rejects_non_psd_measurement_noise():
    kalman_filter = _one_dimensional_filter()

    with pytest.raises(ValueError, match="meas_noise must be positive semidefinite"):
        kalman_filter.innovation_linear(
            backend.array([0.0]),
            backend.array([[1.0]]),
            backend.array([[-0.5]]),
        )


def test_robust_update_rejects_non_psd_measurement_noise_atomically():
    kalman_filter = _one_dimensional_filter()

    with pytest.raises(ValueError, match="meas_noise must be positive semidefinite"):
        kalman_filter.update_linear_robust(
            backend.array([0.0]),
            backend.array([[1.0]]),
            backend.array([[-0.5]]),
        )

    _assert_unit_state_unchanged(kalman_filter)


def test_singular_noise_covariances_remain_supported():
    kalman_filter = _one_dimensional_filter()

    kalman_filter.predict_identity(backend.array([[0.0]]))
    kalman_filter.update_identity(
        backend.array([[0.0]]),
        backend.array([0.0]),
    )

    np.testing.assert_allclose(backend.to_numpy(kalman_filter.filter_state.C), [[0.0]])
