from __future__ import annotations

from pyrecest.tracking import (
    ResidualEditCandidate,
    ResidualMHTConfig,
    enumerate_residual_hypotheses,
    residual_mht_config_for_preset,
    select_residual_hypothesis,
)


def test_selects_best_compatible_residual_hypothesis():
    candidates = [
        ResidualEditCandidate("a", 2.0, frozenset({"target:1"})),
        ResidualEditCandidate("b", 1.8, frozenset({"target:2"})),
        ResidualEditCandidate("c", 3.0, frozenset({"target:1"})),
    ]

    selected = select_residual_hypothesis(
        candidates,
        config=ResidualMHTConfig(max_edits=2, edit_penalty=0.1, score_threshold=0.0),
    )

    assert selected.candidate_ids == ("c", "b")
    assert selected.n_edits == 2


def test_returns_no_edit_below_threshold():
    selected = select_residual_hypothesis(
        [ResidualEditCandidate("weak", 0.5)],
        config=ResidualMHTConfig(max_edits=1, edit_penalty=0.0, score_threshold=1.0),
    )

    assert selected.candidate_ids == ()
    assert selected.score == 0.0


def test_enumerates_no_edit_when_requested():
    hypotheses = enumerate_residual_hypotheses(
        [ResidualEditCandidate("a", 1.0)],
        config=ResidualMHTConfig(max_edits=1, include_empty=True),
    )

    assert any(hypothesis.candidate_ids == () for hypothesis in hypotheses)


def test_conflicting_candidates_are_not_combined():
    hypotheses = enumerate_residual_hypotheses(
        [
            ResidualEditCandidate("a", 1.0, frozenset({"x"})),
            ResidualEditCandidate("b", 1.0, frozenset({"x"})),
        ],
        config=ResidualMHTConfig(max_edits=2, include_empty=False),
    )

    assert all(len(hypothesis.candidate_ids) == 1 for hypothesis in hypotheses)


def test_family_caps_prevent_overselecting_one_candidate_family():
    candidates = [
        ResidualEditCandidate("growth", 2.0, family="growth_veto"),
        ResidualEditCandidate("cell_a", 1.9, family="cell_gated"),
        ResidualEditCandidate("cell_b", 1.8, family="cell_gated"),
    ]

    selected = select_residual_hypothesis(
        candidates,
        config=ResidualMHTConfig(
            max_edits=3,
            edit_penalty=0.0,
            score_threshold=0.0,
            max_edits_per_family={"cell_gated": 1},
        ),
    )

    assert selected.candidate_ids == ("growth", "cell_a")
    assert selected.candidate_families == ("growth_veto", "cell_gated")


def test_hypothesis_dict_serializes_candidate_families():
    hypotheses = enumerate_residual_hypotheses(
        [
            ResidualEditCandidate("a", 1.0, family="alpha"),
            ResidualEditCandidate("b", 0.9, family="beta"),
        ],
        config=ResidualMHTConfig(max_edits=2, edit_penalty=0.0, include_empty=False),
    )

    serialized = [
        {
            "candidate_ids": ";".join(hypothesis.candidate_ids),
            "families": ";".join(hypothesis.candidate_families),
        }
        for hypothesis in hypotheses
    ]

    assert {"candidate_ids": "a;b", "families": "alpha;beta"} in serialized


def test_candidate_top_k_limits_frontier_before_hypothesis_enumeration():
    hypotheses = enumerate_residual_hypotheses(
        [
            ResidualEditCandidate("a", 4.0),
            ResidualEditCandidate("b", 3.0),
            ResidualEditCandidate("c", 2.0),
        ],
        config=ResidualMHTConfig(
            max_edits=3,
            edit_penalty=0.0,
            candidate_top_k=2,
            include_empty=False,
        ),
    )

    assert hypotheses[0].candidate_ids == ("a", "b")
    assert all("c" not in hypothesis.candidate_ids for hypothesis in hypotheses)


def test_min_candidate_score_filters_weak_candidates():
    hypotheses = enumerate_residual_hypotheses(
        [
            ResidualEditCandidate("strong", 2.0),
            ResidualEditCandidate("weak", 0.1),
        ],
        config=ResidualMHTConfig(
            max_edits=2,
            edit_penalty=0.0,
            min_candidate_score=1.0,
            include_empty=False,
        ),
    )

    assert all("weak" not in hypothesis.candidate_ids for hypothesis in hypotheses)


def test_group_key_caps_are_softer_than_conflict_keys():
    candidates = [
        ResidualEditCandidate("a", 3.0, group_keys=frozenset({"component:1"})),
        ResidualEditCandidate("b", 2.0, group_keys=frozenset({"component:1"})),
        ResidualEditCandidate("c", 1.0, group_keys=frozenset({"component:2"})),
    ]

    selected = select_residual_hypothesis(
        candidates,
        config=ResidualMHTConfig(
            max_edits=3,
            edit_penalty=0.0,
            score_threshold=0.0,
            max_edits_per_group_key=1,
        ),
    )

    assert selected.candidate_ids == ("a", "c")
    assert selected.candidate_group_keys == ("component:1", "component:2")


def test_residual_mht_presets_are_generic_selector_breadth_controls():
    conservative = residual_mht_config_for_preset("conservative")
    frontier = residual_mht_config_for_preset("frontier")
    multi_family = residual_mht_config_for_preset("multi_family", max_edits=4)

    assert conservative.max_edits == 1
    assert conservative.candidate_top_k == 4
    assert frontier.max_edits == 2
    assert frontier.candidate_top_k == 8
    assert multi_family.max_edits == 4
    assert multi_family.max_hypotheses == 64
