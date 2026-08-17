"""Regression tests for scale-safe sparse Gaussian grid transitions."""

import numpy as np
from pyrecest.filters.discrete_state import sparse_gaussian_transition_matrix


def test_extreme_finite_grid_coordinates_preserve_gaussian_transition_ratios():
    largest = np.finfo(float).max
    grid = np.array([-largest, largest])

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        transition = sparse_gaussian_transition_matrix(
            grid,
            sigma=largest,
            max_step_sigma=np.inf,
        ).toarray()

    off_diagonal_weight = np.exp(-2.0)
    expected = np.array(
        [
            [1.0, off_diagonal_weight],
            [off_diagonal_weight, 1.0],
        ]
    )
    expected /= 1.0 + off_diagonal_weight

    assert np.all(np.isfinite(transition))
    np.testing.assert_allclose(transition, expected, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(transition.sum(axis=0), np.ones(2))
