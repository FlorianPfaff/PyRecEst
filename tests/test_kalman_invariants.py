import numpy as np
import pyrecest.backend as backend
from pyrecest.backend import array, diag
from pyrecest.filters import KalmanFilter


def _as_numpy(value):
    return np.asarray(backend.to_numpy(value), dtype=float)


def _assert_symmetric_positive_semidefinite(matrix, *, atol=1e-10):
    matrix = _as_numpy(matrix)
    assert np.allclose(matrix, matrix.T, atol=atol)
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    assert np.min(eigenvalues) >= -atol


def test_kalman_predict_update_preserves_covariance_invariants():
    kf = KalmanFilter((array([0.0, 1.0]), diag(array([1.0, 1.0]))))
    system_matrix = array([[1.0, 1.0], [0.0, 1.0]])
    measurement_matrix = array([[1.0, 0.0]])
    system_noise_cov = diag(array([0.05, 0.01]))
    measurement_noise_cov = array([[0.25]])

    kf.predict_linear(system_matrix, system_noise_cov)
    predicted_covariance = _as_numpy(kf.filter_state.C)
    _assert_symmetric_positive_semidefinite(predicted_covariance)

    kf.update_linear(array([0.9]), measurement_matrix, measurement_noise_cov)
    posterior_covariance = _as_numpy(kf.filter_state.C)
    _assert_symmetric_positive_semidefinite(posterior_covariance)

    # A scalar position measurement should not increase total covariance trace
    # for this linear Gaussian example.
    assert np.trace(posterior_covariance) <= np.trace(predicted_covariance) + 1e-10
