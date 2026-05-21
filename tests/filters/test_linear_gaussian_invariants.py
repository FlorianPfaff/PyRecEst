"""Numerical invariants for linear-Gaussian Kalman operations."""

from __future__ import annotations

import numpy as np
from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import KalmanFilter


def _to_numpy(value):
    """Convert backend arrays or scalars to NumPy for backend-independent assertions."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _as_float(value) -> float:
    return float(_to_numpy(value))


def test_kalman_predict_update_preserve_covariance_invariants():
    initial_state = GaussianDistribution(
        array([0.0, 1.0]),
        array([[2.0, 0.2], [0.2, 0.5]]),
        check_validity=False,
    )
    kf = KalmanFilter(initial_state)

    kf.predict_linear(
        array([[1.0, 0.1], [0.0, 1.0]]),
        array([[0.05, 0.0], [0.0, 0.02]]),
    )
    diagnostics = kf.update_linear(
        array([0.2]),
        array([[1.0, 0.0]]),
        array([[0.25]]),
        return_diagnostics=True,
    )

    covariance = _to_numpy(kf.filter_state.C)

    assert np.allclose(covariance, covariance.T, atol=1e-10)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-10
    assert diagnostics["action"] == "updated"
    assert _as_float(diagnostics["nis"]) >= 0.0


def test_normalized_innovation_squared_matches_manual_solve():
    kf = KalmanFilter(
        GaussianDistribution(
            array([1.0, -0.5]),
            array([[1.5, 0.1], [0.1, 0.75]]),
            check_validity=False,
        )
    )
    measurement = array([0.4, -0.1])
    measurement_matrix = array([[1.0, 0.0], [0.0, 1.0]])
    measurement_noise = array([[0.2, 0.03], [0.03, 0.4]])

    innovation, innovation_covariance = kf.innovation_linear(
        measurement,
        measurement_matrix,
        measurement_noise,
    )
    observed = kf.normalized_innovation_squared_linear(
        measurement,
        measurement_matrix,
        measurement_noise,
    )

    innovation_np = _to_numpy(innovation)
    innovation_covariance_np = _to_numpy(innovation_covariance)
    expected = innovation_np.T @ np.linalg.solve(
        innovation_covariance_np, innovation_np
    )

    assert np.allclose(_as_float(observed), expected, atol=1e-10)


def test_rejected_linear_robust_update_leaves_state_unchanged():
    kf = KalmanFilter(
        GaussianDistribution(
            array([0.0]),
            array([[1.0]]),
            check_validity=False,
        )
    )
    prior_mean = _to_numpy(kf.filter_state.mu).copy()
    prior_covariance = _to_numpy(kf.filter_state.C).copy()

    diagnostics = kf.update_linear_robust(
        array([10.0]),
        array([[1.0]]),
        array([[0.1]]),
        robust_update=None,
        gate_threshold=1.0,
        return_diagnostics=True,
    )

    assert diagnostics["accepted"] is False
    assert diagnostics["action"] == "rejected"
    assert np.allclose(_to_numpy(kf.filter_state.mu), prior_mean)
    assert np.allclose(_to_numpy(kf.filter_state.C), prior_covariance)
