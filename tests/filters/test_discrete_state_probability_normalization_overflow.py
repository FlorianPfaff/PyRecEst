import numpy as np

from pyrecest.filters.discrete_state import (
    discrete_forward_backward,
    imm_forward_backward,
)


def test_forward_backward_normalizes_finite_priors_without_overflow():
    huge = np.finfo(float).max
    initial = np.array([huge, huge / 2.0])

    result = discrete_forward_backward(
        np.zeros((1, 2)),
        np.eye(2),
        initial_probabilities=initial,
    )

    np.testing.assert_allclose(
        result.filtered_probabilities[0],
        np.array([2.0 / 3.0, 1.0 / 3.0]),
    )


def test_imm_normalizes_finite_state_and_mode_priors_without_overflow():
    huge = np.finfo(float).max
    state_prior = np.array([huge, huge / 2.0])
    mode_prior = np.array([huge / 2.0, huge])

    result = imm_forward_backward(
        np.zeros((1, 2)),
        [np.eye(2), np.eye(2)],
        np.eye(2),
        initial_state_probabilities=state_prior,
        initial_mode_probabilities=mode_prior,
    )

    expected_state = np.array([2.0 / 3.0, 1.0 / 3.0])
    expected_mode = np.array([1.0 / 3.0, 2.0 / 3.0])
    np.testing.assert_allclose(
        result.filtered_joint_probabilities[0],
        expected_mode[:, None] * expected_state[None, :],
    )
