from __future__ import annotations

import numpy as np
from pyrecest.sampling import mahalanobis_support_points


def test_mahalanobis_support_points_preserve_world_direction() -> None:
    mean = np.asarray([1.0, -2.0])
    covariance = np.diag([4.0, 1.0])
    direction = np.asarray([1.0, 1.0])

    support = mahalanobis_support_points(mean, covariance, direction)

    offset = support[0] - mean
    expected_offset = np.full(2, 2.0 / np.sqrt(5.0))
    assert np.allclose(offset, expected_offset)
    assert np.allclose(offset[0] * direction[1], offset[1] * direction[0])
    assert np.isclose(offset @ np.linalg.solve(covariance, offset), 1.0)


def test_mahalanobis_support_points_normalizes_extreme_finite_direction() -> None:
    mean = np.zeros(2)
    covariance = np.eye(2)
    direction = np.full(2, np.finfo(np.float64).max)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        support = mahalanobis_support_points(mean, covariance, direction)

    expected = np.full(2, 1.0 / np.sqrt(2.0))
    assert np.allclose(support[0], expected)
    assert np.isclose(support[0] @ np.linalg.solve(covariance, support[0]), 1.0)
