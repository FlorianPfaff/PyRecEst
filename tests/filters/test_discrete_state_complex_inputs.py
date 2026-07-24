import unittest

import numpy as np
from pyrecest.filters.discrete_state import (
    discrete_forward_backward,
    discrete_forward_backward_time_varying,
    imm_forward_backward,
    probabilities_to_log_probabilities,
    scaled_emissions,
)
from scipy.sparse import csr_matrix


class TestDiscreteStateComplexInputs(unittest.TestCase):
    def test_probability_utilities_reject_complex_values(self):
        invalid_log_likelihoods = (
            np.array([[0.0 + 0.25j, -1.0 + 0.0j]]),
            np.array([[0.0 + 0.25j, -1.0]], dtype=object),
        )
        for log_likelihood in invalid_log_likelihoods:
            with self.subTest(function="scaled_emissions", value=log_likelihood):
                with self.assertRaisesRegex(ValueError, "real values"):
                    scaled_emissions(log_likelihood)

        invalid_probabilities = (
            np.array([[0.5 + 0.25j, 0.5 + 0.0j]]),
            np.array([[0.5 + 0.25j, 0.5]], dtype=object),
        )
        for probabilities in invalid_probabilities:
            with self.subTest(
                function="probabilities_to_log_probabilities", value=probabilities
            ):
                with self.assertRaisesRegex(ValueError, "real probability values"):
                    probabilities_to_log_probabilities(probabilities, axis=1)

    def test_forward_backward_rejects_complex_emissions(self):
        log_likelihood = np.array([[0.0 + 0.25j, -1.0], [-0.5, -0.25]])

        with self.assertRaisesRegex(ValueError, "log_likelihood.*real values"):
            discrete_forward_backward(log_likelihood, np.eye(2))

    def test_forward_backward_rejects_complex_transition_matrices(self):
        log_likelihood = np.zeros((2, 2), dtype=float)
        complex_transition = np.array([[0.8 + 0.25j, 0.2], [0.2, 0.8]], dtype=complex)

        for transition in (complex_transition, csr_matrix(complex_transition)):
            with self.subTest(transition_type=type(transition).__name__):
                with self.assertRaisesRegex(
                    ValueError, "transition.*real transition probabilities"
                ):
                    discrete_forward_backward(log_likelihood, transition)

    def test_time_varying_forward_backward_rejects_complex_transition(self):
        log_likelihood = np.zeros((2, 2), dtype=float)
        complex_transition = np.array([[0.8 + 0.25j, 0.2], [0.2, 0.8]], dtype=complex)

        with self.assertRaisesRegex(
            ValueError, r"transitions\[0\].*real transition probabilities"
        ):
            discrete_forward_backward_time_varying(log_likelihood, [complex_transition])

    def test_imm_rejects_complex_state_and_mode_transitions(self):
        log_likelihood = np.zeros((2, 2), dtype=float)
        complex_transition = np.array([[0.8 + 0.25j, 0.2], [0.2, 0.8]], dtype=complex)

        with self.assertRaisesRegex(
            ValueError, r"state_transitions\[0\].*real transition probabilities"
        ):
            imm_forward_backward(
                log_likelihood,
                [complex_transition, np.eye(2)],
                np.eye(2),
            )

        complex_mode_transition = np.array(
            [[0.9 + 0.25j, 0.1], [0.1, 0.9]], dtype=complex
        )
        with self.assertRaisesRegex(
            ValueError, "mode_transition.*real transition probabilities"
        ):
            imm_forward_backward(
                log_likelihood,
                [np.eye(2), np.eye(2)],
                complex_mode_transition,
            )


if __name__ == "__main__":
    unittest.main()
