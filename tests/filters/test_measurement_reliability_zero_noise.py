import numpy as np
from pyrecest.filters.measurement_reliability import (
    normalize_measurement_noise_covariances,
)


def _as_covariance_matrix(value, measurement_dim, name):
    covariance = np.asarray(value, dtype=float)
    if covariance.shape != (measurement_dim, measurement_dim):
        raise ValueError(
            f"{name} must have shape ({measurement_dim}, {measurement_dim})"
        )
    return covariance


def test_zero_measurements_with_shared_noise_returns_empty_covariance_batch():
    covariances = normalize_measurement_noise_covariances(
        np.eye(2),
        0,
        2,
        as_covariance_matrix=_as_covariance_matrix,
    )

    assert covariances.shape == (0, 2, 2)


def test_zero_measurements_with_empty_per_measurement_noise_returns_empty_batch():
    covariances = normalize_measurement_noise_covariances(
        np.empty((0, 3, 3)),
        0,
        3,
        as_covariance_matrix=_as_covariance_matrix,
    )

    assert covariances.shape == (0, 3, 3)
