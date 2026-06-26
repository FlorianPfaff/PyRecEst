"""Validation tests for track-edit what-if selector indices."""

import pytest
from pyrecest.utils.track_edit_whatif import (
    TrackEdit,
    apply_track_edit,
    score_track_edit_delta,
)


def test_add_link_rejects_invalid_link_fields() -> None:
    with pytest.raises(
        ValueError, match="target_observation must be a non-negative integer"
    ):
        apply_track_edit(
            [[1, None]],
            TrackEdit(
                kind="add_link",
                session_a=0,
                session_b=1,
                source_observation=1,
                target_observation=2.5,
            ),
        )

    with pytest.raises(ValueError, match="session_a must be a non-negative integer"):
        apply_track_edit(
            [[1, None]],
            TrackEdit(
                kind="add_link",
                session_a=bool(0),
                session_b=1,
                source_observation=1,
                target_observation=2,
            ),
        )


def test_score_rejects_invalid_session_selectors() -> None:
    edit = TrackEdit(
        kind="add_link",
        session_a=0,
        session_b=1,
        source_observation=1,
        target_observation=2,
    )

    with pytest.raises(
        ValueError, match="complete_session_indices must be a non-negative integer"
    ):
        score_track_edit_delta(
            [[1, 2]], [[1, 2]], edit, complete_session_indices=[0, 1.5]
        )

    with pytest.raises(
        ValueError, match="session_pairs must be a non-negative integer"
    ):
        score_track_edit_delta([[1, 2]], [[1, 2]], edit, session_pairs=[(bool(0), 1)])
