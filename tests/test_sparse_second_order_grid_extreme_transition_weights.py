import numpy as np

from pyrecest.filters.sparse_second_order_grid import (
    sparse_second_order_grid_evidence,
)


def test_extreme_finite_transition_weights_preserve_relative_mass():
    maximum = np.finfo(float).max

    def initial_pair_initializer(_scaled_emissions):
        return (
            np.array([0]),
            np.array([0]),
            np.array([1.0]),
            np.array([1]),
        )

    def transition_row_builder(_previous, _current, _transition_index):
        return np.array([0, 1]), np.array([maximum, maximum / 2.0])

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = sparse_second_order_grid_evidence(
            np.zeros((3, 2)),
            initial_pair_initializer,
            transition_row_builder,
            return_smoothed=False,
        )

    np.testing.assert_allclose(
        np.exp(result.terminal_log_probabilities),
        np.array([2.0 / 3.0, 1.0 / 3.0]),
        rtol=1e-14,
        atol=0.0,
    )
