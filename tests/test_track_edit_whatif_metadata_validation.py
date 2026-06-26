"""Regression tests for track-edit metadata selector validation."""

import pytest
from pyrecest.utils.track_edit_whatif import TrackEdit, apply_track_edit


def test_remove_link_rejects_fractional_occurrence_index() -> None:
    edit = TrackEdit(
        kind="remove_link",
        session_a=0,
        session_b=1,
        source_observation=1,
        target_observation=2,
        metadata={"occurrence_index": 0.5},
    )

    with pytest.raises(
        ValueError, match="metadata\['occurrence_index'\] must be an integer"
    ):
        apply_track_edit([[1, 2], [1, 2]], edit)


def test_swap_link_rejects_boolean_removed_observation() -> None:
    edit = TrackEdit(
        kind="swap_link",
        session_a=0,
        session_b=1,
        source_observation=1,
        target_observation=3,
        metadata={
            "remove_source_observation": bool(1),
            "remove_target_observation": 2,
        },
    )

    with pytest.raises(
        ValueError,
        match="metadata\['remove_source_observation'\] must be a non-negative integer",
    ):
        apply_track_edit([[1, 2]], edit)


def test_merge_tracks_rejects_text_other_track_index() -> None:
    edit = TrackEdit(
        kind="merge_tracks",
        track_index=0,
        metadata={"other_track_index": "1"},
    )

    with pytest.raises(
        ValueError, match="metadata\['other_track_index'\] must be an integer"
    ):
        apply_track_edit([[1, None], [None, 2]], edit)
