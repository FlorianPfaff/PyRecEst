from __future__ import annotations

import pytest
from pyrecest.tracking import (
    ResidualEditCandidate,
    ResidualMHTConfig,
    enumerate_residual_hypotheses,
)


def test_duplicate_candidate_ids_are_deduplicated_before_frontier_limits():
    hypotheses = enumerate_residual_hypotheses(
        [
            ResidualEditCandidate("a", 5.0),
            ResidualEditCandidate("a", 4.0),
            ResidualEditCandidate("b", 3.0),
        ],
        config=ResidualMHTConfig(
            max_edits=2,
            max_hypotheses=8,
            edit_penalty=0.0,
            candidate_top_k=2,
            include_empty=False,
        ),
    )

    assert hypotheses[0].candidate_ids == ("a", "b")
    assert hypotheses[0].score == 8.0
    assert all(
        len(hypothesis.candidate_ids) == len(set(hypothesis.candidate_ids))
        for hypothesis in hypotheses
    )
    assert {hypothesis.candidate_ids for hypothesis in hypotheses} == {
        ("a", "b"),
        ("a",),
        ("b",),
    }


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_residual_edit_candidate_rejects_nonfinite_scores(score):
    with pytest.raises(ValueError, match="score must be finite"):
        ResidualEditCandidate("invalid", score)
