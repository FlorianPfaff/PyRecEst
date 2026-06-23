import itertools
import unittest

import numpy as np
from pyrecest.filters.discrete_state import (
    discrete_forward_backward,
    discrete_forward_backward_time_varying,
    imm_forward_backward,
    mode_transition_matrix,
    probabilities_to_log_probabilities,
    scaled_emissions,
    sparse_gaussian_transition_matrix,
    sticky_mode_transition_matrix,
    uniform_probabilities,
)
from scipy.sparse import csr_matrix


def _enumerate_hmm(log_likelihood, transition, initial):
    likelihood = np.exp(log_likelihood)
    n_time, n_states = likelihood.shape
    path_probabilities = {}
    log_total = -np.inf
    smoothed = np.zeros((n_time, n_states), dtype=float)
    for path in itertools.product(range(n_states), repeat=n_time):
        prob = initial[path[0]] * likelihood[0, path[0]]
        for time_index in range(1, n_time):
            prob *= (
                transition[path[time_index], path[time_index - 1]]
                * likelihood[time_index, path[time_index]]
            )
        path_probabilities[path] = prob
        if prob > 0.0:
            log_total = np.logaddexp(log_total, np.log(prob))
        for time_index, state in enumerate(path):
            smoothed[time_index, state] += prob
    total = sum(path_probabilities.values())
    smoothed /= total
    return log_total, smoothed


class TestDiscreteStateUtilities(unittest.TestCase):
    def test_scaled_emissions_uses_row_offsets_and_zeros_nonfinite_entries(self):
        log_likelihood = np.array(
            [
                [-1000.0, -1001.0, -np.inf],
                [3.0, 1.0, 2.0],
            ]
        )

        scaled, offsets = scaled_emissions(log_likelihood)

        np.testing.assert_allclose(offsets, np.array([-1000.0, 3.0]))
        np.testing.assert_allclose(scaled[0], np.array([1.0, np.exp(-1.0), 0.0]))
        np.testing.assert_allclose(
            scaled[1], np.array([1.0, np.exp(-2.0), np.exp(-1.0)])
        )

    def test_probabilities_to_log_probabilities_preserves_zero_support(self):
        logs = probabilities_to_log_probabilities(np.array([[0.2, 0.0, 0.3]]), axis=1)

        self.assertLess(logs[0, 1], -1.0e100)
        np.testing.assert_allclose(np.exp(logs[0, [0, 2]]), np.array([0.4, 0.6]))

    def test_forward_backward_matches_exhaustive_path_enumeration(self):
        log_likelihood = np.log(
            np.array(
                [
                    [0.8, 0.3],
                    [0.4, 0.9],
                    [0.7, 0.2],
                ]
            )
        )
        transition = np.array(
            [
                [0.85, 0.15],
                [0.15, 0.85],
            ]
        )
        initial = np.array([0.6, 0.4])

        result = discrete_forward_backward(
            log_likelihood, csr_matrix(transition), initial_probabilities=initial
        )
        expected_log_evidence, expected_smoothed = _enumerate_hmm(
            log_likelihood, transition, initial
        )

        self.assertAlmostEqual(result.log_marginal_likelihood, expected_log_evidence)
        np.testing.assert_allclose(result.smoothed_probabilities, expected_smoothed)
        np.testing.assert_allclose(
            result.filtered_probabilities.sum(axis=1), np.ones(log_likelihood.shape[0])
        )
        np.testing.assert_allclose(
            result.smoothed_probabilities.sum(axis=1), np.ones(log_likelihood.shape[0])
        )

    def test_time_varying_forward_backward_matches_constant_version(self):
        log_likelihood = np.log(np.array([[0.5, 0.2], [0.1, 0.8], [0.9, 0.4]]))
        transition = csr_matrix(np.array([[0.7, 0.2], [0.3, 0.8]]))

        constant_result = discrete_forward_backward(log_likelihood, transition)
        time_varying_result = discrete_forward_backward_time_varying(
            log_likelihood, [transition, transition]
        )

        self.assertAlmostEqual(
            time_varying_result.log_marginal_likelihood,
            constant_result.log_marginal_likelihood,
        )
        np.testing.assert_allclose(
            time_varying_result.smoothed_probabilities,
            constant_result.smoothed_probabilities,
        )

    def test_sparse_gaussian_transition_matrix_is_column_stochastic_and_respects_mask(
        self,
    ):
        grid = np.array([[0.0], [1.0], [2.0], [3.0]])
        valid = np.array([True, False, True, True])

        transition = sparse_gaussian_transition_matrix(
            grid, sigma=1.0, max_step_sigma=1.5, valid_state_mask=valid
        )
        dense = transition.toarray()

        np.testing.assert_allclose(dense.sum(axis=0), np.ones(grid.shape[0]))
        np.testing.assert_allclose(dense[~valid], np.zeros((1, grid.shape[0])))
        self.assertGreater(transition.nnz, 0)

    def test_sticky_mode_transition_matrix_is_row_stochastic(self):
        matrix = sticky_mode_transition_matrix(3, stickiness=0.8)

        np.testing.assert_allclose(matrix.sum(axis=1), np.ones(3))
        np.testing.assert_allclose(np.diag(matrix), np.full(3, 0.8))
        np.testing.assert_allclose(mode_transition_matrix(2, 0.5), np.full((2, 2), 0.5))

    def test_imm_forward_backward_reduces_to_single_mode_hmm(self):
        log_likelihood = np.log(np.array([[0.9, 0.2], [0.1, 0.8], [0.6, 0.4]]))
        transition = csr_matrix(np.array([[0.75, 0.25], [0.25, 0.75]]))
        initial = np.array([0.55, 0.45])

        hmm_result = discrete_forward_backward(
            log_likelihood, transition, initial_probabilities=initial
        )
        imm_result = imm_forward_backward(
            log_likelihood,
            [transition],
            np.ones((1, 1)),
            initial_state_probabilities=initial,
        )

        self.assertAlmostEqual(
            imm_result.log_marginal_likelihood, hmm_result.log_marginal_likelihood
        )
        np.testing.assert_allclose(
            imm_result.smoothed_state_probabilities, hmm_result.smoothed_probabilities
        )
        np.testing.assert_allclose(
            imm_result.smoothed_mode_probabilities,
            np.ones((log_likelihood.shape[0], 1)),
        )

    def test_imm_reset_transition_keeps_valid_state_support_normalized(self):
        log_likelihood = np.log(np.array([[0.5, 0.1, 0.2], [0.2, 0.7, 0.1]]))
        valid = np.array([True, False, True])
        mode_matrix = sticky_mode_transition_matrix(2, 0.9)

        result = imm_forward_backward(
            log_likelihood,
            [csr_matrix(np.eye(3)), None],
            mode_matrix,
            valid_state_mask=valid,
        )

        np.testing.assert_allclose(
            result.filtered_joint_probabilities.sum(axis=(1, 2)), np.ones(2)
        )
        np.testing.assert_allclose(
            result.smoothed_joint_probabilities.sum(axis=(1, 2)), np.ones(2)
        )
        np.testing.assert_allclose(
            result.smoothed_state_probabilities[:, ~valid], np.zeros((2, 1))
        )

    def test_uniform_probabilities_honors_valid_mask(self):
        probs = uniform_probabilities(4, np.array([False, True, True, False]))

        np.testing.assert_allclose(probs, np.array([0.0, 0.5, 0.5, 0.0]))

    def test_valid_state_mask_rejects_numeric_inputs(self):
        invalid_masks = (
            [1.0, 0.0, 1.0],
            np.array([1, 0, 1]),
            np.array([1.0, 2.0, 3.0]),
        )

        for invalid_mask in invalid_masks:
            with self.subTest(invalid_mask=invalid_mask):
                with self.assertRaisesRegex(ValueError, "boolean"):
                    uniform_probabilities(3, invalid_mask)


if __name__ == "__main__":
    unittest.main()
