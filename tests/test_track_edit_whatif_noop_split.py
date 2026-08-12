"""Regression tests for no-op split-track edits."""

from pyrecest.utils.track_edit_whatif import (
    TrackEdit,
    apply_track_edit,
    score_track_edit_delta,
)


def test_split_track_rejects_empty_split_side() -> None:
    cases = (
        ([[None, None, 7, 8]], 1),
        ([[7, 8, None, None]], 3),
    )

    for predicted, split_session in cases:
        edit = TrackEdit(
            kind="split_track",
            track_index=0,
            session_b=split_session,
        )

        application = apply_track_edit(predicted, edit)
        delta = score_track_edit_delta(predicted, predicted, edit)

        assert not application.applied
        assert application.action == "reject"
        assert application.reason == "empty_split_side"
        assert application.track_matrix.tolist() == predicted
        assert not delta.applied
        assert delta.reason == "empty_split_side"
        assert delta.pairwise_tp_delta == 0
        assert delta.pairwise_fp_delta == 0
        assert delta.pairwise_fn_delta == 0
        assert delta.complete_tp_delta == 0
        assert delta.complete_fp_delta == 0
        assert delta.complete_fn_delta == 0


def test_split_track_still_applies_when_both_sides_are_nonempty() -> None:
    predicted = [[1, 2, 3, 4]]
    edit = TrackEdit(kind="split_track", track_index=0, session_b=2)

    application = apply_track_edit(predicted, edit)

    assert application.applied
    assert application.action == "split_track"
    assert application.reason == "accepted"
    assert application.track_matrix.tolist() == [
        [1, 2, None, None],
        [None, None, 3, 4],
    ]
