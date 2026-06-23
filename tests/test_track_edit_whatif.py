"""Tests for generic track-edit what-if scoring."""

from pyrecest.utils.track_edit_whatif import (
    TrackEdit,
    apply_track_edit,
    rank_track_edits_by_delta,
    score_track_edit_delta,
)


def test_add_link_scores_clean_completion_delta() -> None:
    predicted = [[1, 2, None]]
    reference = [[1, 2, 3]]
    edit = TrackEdit(
        kind="add_link",
        session_a=1,
        session_b=2,
        source_observation=2,
        target_observation=3,
    )

    delta = score_track_edit_delta(predicted, reference, edit)

    assert delta.applied
    assert delta.action == "insert_target"
    assert delta.pairwise_tp_delta == 1
    assert delta.pairwise_fn_delta == -1
    assert delta.complete_tp_delta == 1
    assert delta.new_complete_track_f1 == 1.0


def test_remove_link_splits_track_and_removes_false_positive() -> None:
    predicted = [[1, 2, 99, 4]]
    reference = [[1, 2, 3, 4]]
    edit = TrackEdit(
        kind="remove_link",
        session_a=1,
        session_b=2,
        source_observation=2,
        target_observation=99,
    )

    application = apply_track_edit(predicted, edit)
    delta = score_track_edit_delta(predicted, reference, edit)

    assert application.applied
    assert application.action == "split_component_at_edge"
    assert application.track_matrix.tolist() == [
        [1, 2, None, None],
        [None, None, 99, 4],
    ]
    assert delta.pairwise_fp_delta == -1
    assert delta.complete_fp_delta == -1


def test_swap_link_replaces_wrong_adjacent_edge() -> None:
    predicted = [[1, 2, 99]]
    reference = [[1, 2, 3]]
    edit = TrackEdit(
        kind="swap_link",
        session_a=1,
        session_b=2,
        source_observation=2,
        target_observation=3,
        metadata={
            "remove_session_a": 1,
            "remove_session_b": 2,
            "remove_source_observation": 2,
            "remove_target_observation": 99,
        },
    )

    delta = score_track_edit_delta(predicted, reference, edit)

    assert delta.applied
    assert delta.action == "swap_adjacent_edge"
    assert delta.pairwise_tp_delta == 1
    assert delta.pairwise_fp_delta == -1
    assert delta.pairwise_fn_delta == -1
    assert delta.complete_tp_delta == 1
    assert delta.complete_fp_delta == -1


def test_add_link_rejects_duplicate_source() -> None:
    predicted = [[1, 2, 99]]
    reference = [[1, 2, 3]]
    edit = TrackEdit(
        kind="add_link",
        session_a=1,
        session_b=2,
        source_observation=2,
        target_observation=3,
    )

    delta = score_track_edit_delta(predicted, reference, edit)

    assert not delta.applied
    assert delta.creates_duplicate_source
    assert delta.reason == "duplicate_source_or_target"
    assert delta.pairwise_tp_delta == 0


def test_duplicate_sensitive_scoring_counts_duplicate_false_positive() -> None:
    predicted = [[1, 2], [1, 2]]
    reference = [[1, 2]]
    edit = TrackEdit(
        kind="remove_link",
        session_a=0,
        session_b=1,
        source_observation=1,
        target_observation=2,
        metadata={"occurrence_index": 1},
    )

    set_delta = score_track_edit_delta(predicted, reference, edit)
    multiset_delta = score_track_edit_delta(
        predicted, reference, edit, count_duplicates=True
    )

    assert set_delta.pairwise_fp_delta == 0
    assert multiset_delta.pairwise_fp_delta == -1


def test_remove_link_rejects_negative_occurrence_index() -> None:
    predicted = [[1, 2], [1, 2]]
    edit = TrackEdit(
        kind="remove_link",
        session_a=0,
        session_b=1,
        source_observation=1,
        target_observation=2,
        metadata={"occurrence_index": -1},
    )

    application = apply_track_edit(predicted, edit)

    assert not application.applied
    assert application.action == "reject"
    assert application.reason == "edge_not_found"
    assert application.track_matrix.tolist() == predicted


def test_rank_track_edits_prefers_complete_track_improvement() -> None:
    predicted = [[1, 2, None], [8, 9, 10]]
    reference = [[1, 2, 3], [8, 9, 10]]
    edits = [
        TrackEdit(
            kind="remove_link",
            session_a=0,
            session_b=1,
            source_observation=8,
            target_observation=9,
        ),
        TrackEdit(
            kind="add_link",
            session_a=1,
            session_b=2,
            source_observation=2,
            target_observation=3,
        ),
    ]

    ranked = rank_track_edits_by_delta(predicted, reference, edits)

    assert ranked[0].edit.kind == "add_link"
    assert ranked[0].complete_tp_delta == 1
