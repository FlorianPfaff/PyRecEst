"""Regression tests for masked discrete-state emission scaling."""

import numpy as np
from pyrecest.filters.discrete_state import (
    discrete_forward_backward,
    discrete_forward_backward_time_varying,
    imm_forward_backward,
)


def _masked_emission_case():
    log_likelihood = np.array(
        [
            [1000.0, 0.0],
            [1000.0, np.log(0.5)],
        ]
    )
    valid_state_mask = np.array([False, True])
    expected_probabilities = np.array([[0.0, 1.0], [0.0, 1.0]])
    expected_offsets = np.array([0.0, np.log(0.5)])
    return (
        log_likelihood,
        valid_state_mask,
        expected_probabilities,
        expected_offsets,
    )


def _assert_masked_result(result, expected_probabilities, expected_offsets):
    np.testing.assert_allclose(
        result.log_marginal_likelihood,
        np.log(0.5),
    )
    np.testing.assert_allclose(result.emission_offsets, expected_offsets)
    np.testing.assert_allclose(result.filtered_probabilities, expected_probabilities)
    np.testing.assert_allclose(result.smoothed_probabilities, expected_probabilities)


def test_valid_state_mask_excludes_invalid_emissions_from_hmm_scaling():
    (
        log_likelihood,
        valid_state_mask,
        expected_probabilities,
        expected_offsets,
    ) = _masked_emission_case()
    transition = np.eye(2)

    constant_result = discrete_forward_backward(
        log_likelihood,
        transition,
        valid_state_mask=valid_state_mask,
    )
    time_varying_result = discrete_forward_backward_time_varying(
        log_likelihood,
        [transition],
        valid_state_mask=valid_state_mask,
    )

    _assert_masked_result(
        constant_result,
        expected_probabilities,
        expected_offsets,
    )
    _assert_masked_result(
        time_varying_result,
        expected_probabilities,
        expected_offsets,
    )


def test_valid_state_mask_excludes_invalid_emissions_from_imm_scaling():
    (
        log_likelihood,
        valid_state_mask,
        expected_probabilities,
        expected_offsets,
    ) = _masked_emission_case()

    result = imm_forward_backward(
        log_likelihood,
        [np.eye(2)],
        np.ones((1, 1)),
        valid_state_mask=valid_state_mask,
    )

    np.testing.assert_allclose(result.log_marginal_likelihood, np.log(0.5))
    np.testing.assert_allclose(result.emission_offsets, expected_offsets)
    np.testing.assert_allclose(
        result.filtered_state_probabilities,
        expected_probabilities,
    )
    np.testing.assert_allclose(
        result.smoothed_state_probabilities,
        expected_probabilities,
    )
