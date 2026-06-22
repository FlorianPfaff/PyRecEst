"""Dense label conversion helpers for multi-session assignments."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Any

from pyrecest.backend import int64

from .multisession_assignment import (  # pylint: disable=protected-access
    SessionSizesInput,
    TrackInput,
    _ensure_supported_backend,
    _full_1d,
    _iter_track_items,
    _validate_track_session_sizes,
)


def _normalize_fill_value(fill_value: Any, track_count: int) -> int:
    if isinstance(fill_value, bool) or not isinstance(fill_value, Integral):
        raise ValueError("fill_value must be an integer.")
    fill_value = int(fill_value)
    if 0 <= fill_value < int(track_count):
        raise ValueError("fill_value must not collide with track labels.")
    return fill_value


def tracks_to_session_labels(
    tracks: Sequence[TrackInput],
    session_sizes: SessionSizesInput | None = None,
    *,
    fill_value: int = -1,
) -> tuple[Any, ...]:
    """Convert explicit tracks to dense per-session label arrays."""
    _ensure_supported_backend("tracks_to_session_labels")
    fill_value = _normalize_fill_value(fill_value, len(tracks))

    inferred_sizes, max_session_index = _validate_track_session_sizes(
        tracks,
        session_sizes,
        require_unique_sessions=True,
    )

    labels = [
        _full_1d(inferred_sizes.get(session_index, 0), fill_value, int64)
        for session_index in range(max_session_index + 1)
    ]

    assigned_observations: set[tuple[int, int]] = set()
    for track_index, track in enumerate(tracks):
        for session_index, detection_index in _iter_track_items(track):
            observation = (session_index, detection_index)
            if observation in assigned_observations:
                raise ValueError("Each detection can only belong to a single track.")
            assigned_observations.add(observation)
            labels[session_index][detection_index] = track_index

    return tuple(labels)
