"""Regression tests for empty split-track edits."""

from pyrecest.utils.track_edit_whatif import (
    TrackEdit,
    apply_track_edit,
    score_track_edit_delta,
)


def test_split_track_rejects_empty_track() -> None:
    predicted = [[None, None, None]]
    edit = TrackEdit(kind="split_track", track_index=0, session_b=1)

    application = apply_track_edit(predicted, edit)
    delta = score_track_edit_delta(predicted, predicted, edit)

    assert not application.applied
    assert application.action == "reject"
    assert application.reason == "empty_track"
    assert application.track_matrix.tolist() == predicted
    assert not delta.applied
    assert delta.reason == "empty_track"
    assert delta.pairwise_tp_delta == 0
