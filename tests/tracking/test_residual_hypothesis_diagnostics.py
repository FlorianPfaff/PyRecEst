from __future__ import annotations

from pyrecest.tracking import (
    ResidualEditCandidate,
    ResidualMHTConfig,
    candidates_to_dicts,
    enumerate_residual_hypotheses,
    hypotheses_to_diagnostic_dicts,
    select_residual_hypothesis,
    selection_ledger_to_dicts,
)


def test_candidates_to_dicts_marks_selected_and_applied_candidates() -> None:
    candidates = [
        ResidualEditCandidate(
            "b",
            1.0,
            conflict_keys=frozenset({"target:2"}),
            metadata={"subject": "jm038"},
            family="cell_gated",
        ),
        ResidualEditCandidate(
            "a",
            2.0,
            conflict_keys=frozenset({"target:1"}),
            metadata={"subject": "jm046"},
            family="growth_veto",
        ),
    ]

    rows = candidates_to_dicts(candidates, selected_ids={"a"}, applied_ids={"a"})

    assert rows[0]["candidate_id"] == "a"
    assert rows[0]["rank"] == 1
    assert rows[0]["selected"] == 1
    assert rows[0]["applied"] == 1
    assert rows[0]["family"] == "growth_veto"
    assert rows[0]["conflict_keys"] == "target:1"
    assert rows[0]["metadata_subject"] == "jm046"
    assert rows[1]["candidate_id"] == "b"
    assert rows[1]["selected"] == 0


def test_hypotheses_to_diagnostic_dicts_marks_selected_hypothesis() -> None:
    candidates = [
        ResidualEditCandidate("a", 2.0, family="growth_veto"),
        ResidualEditCandidate("b", 1.0, family="cell_gated"),
    ]
    config = ResidualMHTConfig(max_edits=2, edit_penalty=0.0, score_threshold=0.0)
    hypotheses = enumerate_residual_hypotheses(candidates, config=config)
    selected = select_residual_hypothesis(candidates, config=config)

    rows = hypotheses_to_diagnostic_dicts(hypotheses, selected_hypothesis=selected)

    assert rows[0]["candidate_ids"] == "a;b"
    assert rows[0]["candidate_families"] == "growth_veto;cell_gated"
    assert rows[0]["selected"] == 1
    assert any(row["selected"] == 0 for row in rows[1:])


def test_selection_ledger_to_dicts_returns_candidate_and_hypothesis_tables() -> None:
    candidates = [
        ResidualEditCandidate("a", 2.0, family="growth_veto"),
        ResidualEditCandidate("b", 1.0, family="cell_gated"),
    ]
    config = ResidualMHTConfig(max_edits=1, edit_penalty=0.0, score_threshold=0.0)
    hypotheses = enumerate_residual_hypotheses(candidates, config=config)
    selected = select_residual_hypothesis(candidates, config=config)

    ledger = selection_ledger_to_dicts(
        candidates,
        hypotheses=hypotheses,
        selected_hypothesis=selected,
        applied_ids={"a"},
    )

    assert set(ledger) == {"candidates", "hypotheses"}
    assert ledger["candidates"][0]["candidate_id"] == "a"
    assert ledger["candidates"][0]["selected"] == 1
    assert ledger["candidates"][0]["applied"] == 1
    assert ledger["hypotheses"][0]["selected"] == 1
