from __future__ import annotations

import numpy as np
from pyrecest.filters import (
    AssociationHypothesis,
    hypotheses_to_cost_matrix,
    hypotheses_to_log_likelihood_matrix,
    hypotheses_to_probability_matrix,
)


def _single_entry(converter, hypotheses, **kwargs) -> float:
    matrix = converter(
        hypotheses,
        num_tracks=1,
        num_measurements=1,
        **kwargs,
    )
    return float(matrix[0, 0])


def test_cost_matrix_keeps_lowest_duplicate_pair_cost_independent_of_order() -> None:
    hypotheses = [
        AssociationHypothesis(0, 0, cost=1.0),
        AssociationHypothesis(0, 0, cost=100.0),
    ]

    assert _single_entry(hypotheses_to_cost_matrix, hypotheses) == 1.0
    assert _single_entry(hypotheses_to_cost_matrix, hypotheses[::-1]) == 1.0


def test_log_likelihood_matrix_keeps_highest_duplicate_pair_value() -> None:
    hypotheses = [
        AssociationHypothesis(0, 0, log_likelihood=-4.0),
        AssociationHypothesis(0, 0, log_likelihood=-1.0),
    ]

    assert _single_entry(hypotheses_to_log_likelihood_matrix, hypotheses) == -1.0
    assert _single_entry(hypotheses_to_log_likelihood_matrix, hypotheses[::-1]) == -1.0


def test_probability_matrix_keeps_highest_duplicate_pair_value() -> None:
    hypotheses = [
        AssociationHypothesis(0, 0, probability=0.2),
        AssociationHypothesis(0, 0, probability=0.8),
    ]

    assert _single_entry(hypotheses_to_probability_matrix, hypotheses) == 0.8
    assert _single_entry(hypotheses_to_probability_matrix, hypotheses[::-1]) == 0.8


def test_finite_duplicate_pair_value_wins_over_nan_independent_of_order() -> None:
    hypotheses = [
        AssociationHypothesis(0, 0, cost=np.nan),
        AssociationHypothesis(0, 0, cost=2.0),
    ]

    assert _single_entry(hypotheses_to_cost_matrix, hypotheses) == 2.0
    assert _single_entry(hypotheses_to_cost_matrix, hypotheses[::-1]) == 2.0


def test_rejected_duplicate_does_not_overwrite_better_accepted_cost() -> None:
    hypotheses = [
        AssociationHypothesis(0, 0, cost=2.0),
        AssociationHypothesis(0, 0, cost=0.0, accepted=False),
    ]

    assert (
        _single_entry(
            hypotheses_to_cost_matrix,
            hypotheses,
            include_rejected=True,
            rejected_cost=99.0,
        )
        == 2.0
    )
    assert (
        _single_entry(
            hypotheses_to_cost_matrix,
            hypotheses[::-1],
            include_rejected=True,
            rejected_cost=99.0,
        )
        == 2.0
    )
