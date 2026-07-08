"""Regression tests for scalar-array candidate session providers."""

from __future__ import annotations

import numpy as np
from pyrecest.utils.track_completion import (
    enumerate_fragment_completion_paths,
    path_observations,
    path_sessions,
)


def test_custom_candidate_sessions_accept_zero_dimensional_integer_arrays() -> None:
    tracks = [[7, None, None]]

    def candidate_sessions(session: int, observation: int, direction: str):
        del session, observation, direction
        return [np.array(2)]

    def provider(session: int, observation: int, target_session: int):
        if (session, observation, target_session) == (0, 7, 2):
            return [9]
        return []

    paths = enumerate_fragment_completion_paths(
        tracks,
        direction="suffix",
        candidate_provider=provider,
        candidate_session_provider=candidate_sessions,
    )

    assert len(paths) == 1
    assert path_sessions(paths[0]) == (0, 2)
    assert path_observations(paths[0]) == (7, 9)
