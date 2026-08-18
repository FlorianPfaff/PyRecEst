import pytest

from pyrecest import backend
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import KalmanFilter


def test_kalman_constructor_rejects_indefinite_covariance():
    with pytest.raises(ValueError, match="positive semidefinite"):
        KalmanFilter(
            (
                backend.array([0.0, 0.0]),
                backend.array([[1.0, 2.0], [2.0, 1.0]]),
            )
        )


def test_kalman_state_setter_rejects_indefinite_covariance_atomically():
    kf = KalmanFilter(
        (
            backend.array([1.0, -1.0]),
            backend.array([[2.0, 0.0], [0.0, 1.0]]),
        )
    )
    prior = kf.filter_state
    invalid_state = GaussianDistribution(
        backend.array([4.0, 5.0]),
        backend.array([[1.0, 2.0], [2.0, 1.0]]),
        check_validity=False,
    )

    with pytest.raises(ValueError, match="positive semidefinite"):
        kf.filter_state = invalid_state

    assert bool(backend.allclose(kf.filter_state.mu, prior.mu))
    assert bool(backend.allclose(kf.filter_state.C, prior.C))


def test_kalman_accepts_singular_positive_semidefinite_covariance():
    covariance = backend.array([[1.0, 1.0], [1.0, 1.0]])
    kf = KalmanFilter((backend.array([0.0, 0.0]), covariance))

    assert bool(backend.allclose(kf.filter_state.C, covariance))
