"""Regression coverage for tiny sparse-transition scales."""

import numpy as np
import numpy.testing as npt

from pyrecest.filters.discrete_state import sparse_gaussian_transition_matrix


def test_sparse_gaussian_transition_handles_subnormal_sigma():
    sigma = np.nextafter(0.0, 1.0)

    with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
        transition = sparse_gaussian_transition_matrix(
            np.array([0.0, 1.0]),
            sigma,
        )

    dense = transition.toarray()
    assert np.all(np.isfinite(transition.data))
    npt.assert_array_equal(dense, np.eye(2))
    npt.assert_allclose(
        np.asarray(transition.sum(axis=0), dtype=float).reshape(-1),
        np.ones(2),
    )
