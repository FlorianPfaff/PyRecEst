"""Tests for generic track-fragment completion utilities."""

from __future__ import annotations

import unittest

import numpy as np
from pyrecest.utils.track_completion import (
    CompletionCandidate,
    enumerate_fragment_completion_paths,
    path_observations,
    path_sessions,
)


class TestTrackCompletion(unittest.TestCase):
    def test_enumerates_suffix_paths_with_scores_and_payloads(self) -> None:
        tracks = [[0, 1, None, None], [4, None, None, None]]

        def provider(session: int, observation: int, target_session: int):
            if (session, observation, target_session) == (1, 1, 2):
                return [CompletionCandidate(2, score=0.5, payload="first")]
            if (session, observation, target_session) == (2, 2, 3):
                return [CompletionCandidate(3, score=0.25, payload="second")]
            return []

        paths = enumerate_fragment_completion_paths(
            tracks,
            max_path_length=2,
            direction="suffix",
            candidate_provider=provider,
        )

        self.assertEqual(len(paths), 2)
        self.assertEqual(path_sessions(paths[0]), (1, 2))
        self.assertEqual(path_observations(paths[0]), (1, 2))
        self.assertEqual(paths[0].score, 0.5)
        self.assertEqual(paths[0].steps[0].payload, "first")
        self.assertEqual(path_sessions(paths[1]), (1, 2, 3))
        self.assertEqual(path_observations(paths[1]), (1, 2, 3))
        self.assertEqual(paths[1].score, 0.75)

    def test_rejects_duplicate_targets_by_default(self) -> None:
        tracks = [[0, 1, None], [None, None, 8]]

        def provider(session: int, observation: int, target_session: int):
            del session, observation, target_session
            return [8]

        paths = enumerate_fragment_completion_paths(
            tracks,
            direction="suffix",
            candidate_provider=provider,
        )
        self.assertEqual(paths, [])

        allowed = enumerate_fragment_completion_paths(
            tracks,
            direction="suffix",
            candidate_provider=provider,
            allow_duplicate_target=True,
        )
        self.assertEqual(len(allowed), 1)

    def test_rejects_negative_candidate_observations(self) -> None:
        tracks = [[0, None]]
        invalid_candidates = (-1, CompletionCandidate(-1))

        for invalid_candidate in invalid_candidates:
            with self.subTest(invalid_candidate=invalid_candidate):

                def provider(session: int, observation: int, target_session: int):
                    del session, observation, target_session
                    return [invalid_candidate]

                with self.assertRaisesRegex(
                    ValueError,
                    "candidate observations must be non-negative integers",
                ):
                    enumerate_fragment_completion_paths(
                        tracks,
                        direction="suffix",
                        candidate_provider=provider,
                    )

    def test_enumerates_prefix_paths(self) -> None:
        tracks = [[None, 5, 6]]

        def provider(session: int, observation: int, target_session: int):
            if (session, observation, target_session) == (1, 5, 0):
                return [CompletionCandidate(4, score=1.5)]
            return []

        paths = enumerate_fragment_completion_paths(
            tracks,
            direction="prefix",
            candidate_provider=provider,
        )

        self.assertEqual(len(paths), 1)
        self.assertEqual(path_sessions(paths[0]), (0, 1))
        self.assertEqual(path_observations(paths[0]), (4, 5))
        self.assertEqual(paths[0].score, 1.5)

    def test_custom_candidate_sessions_allow_skip_completion(self) -> None:
        tracks = [[7, None, None]]
        calls: list[tuple[int, int, str]] = []

        def candidate_sessions(session: int, observation: int, direction: str):
            calls.append((session, observation, direction))
            return [2]

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

        self.assertEqual(len(paths), 1)
        self.assertEqual(path_sessions(paths[0]), (0, 2))
        self.assertEqual(path_observations(paths[0]), (7, 9))
        self.assertEqual(calls[0], (0, 7, "suffix"))
        self.assertEqual(calls[1], (2, 9, "suffix"))

    def test_rejects_fractional_integer_controls(self) -> None:
        tracks = [[0, None, None]]

        def provider(session: int, observation: int, target_session: int):
            del session, observation, target_session
            return [1]

        invalid_kwargs = (
            {"max_path_length": 1.5},
            {"max_paths_per_fragment": 1.5},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "must be a positive integer"):
                    enumerate_fragment_completion_paths(
                        tracks,
                        direction="suffix",
                        candidate_provider=provider,
                        **kwargs,
                    )

    def test_custom_candidate_sessions_do_not_truncate_invalid_scalars(self) -> None:
        tracks = [[7, None, None]]

        def candidate_sessions(session: int, observation: int, direction: str):
            del session, observation, direction
            return [True, np.bool_(True), 1.5, np.float64(1.5)]

        def provider(session: int, observation: int, target_session: int):
            if (session, observation, target_session) == (0, 7, 1):
                return [8]
            return []

        paths = enumerate_fragment_completion_paths(
            tracks,
            direction="suffix",
            candidate_provider=provider,
            candidate_session_provider=candidate_sessions,
        )

        self.assertEqual(paths, [])

    def test_rejects_invalid_path_limit_scalars(self) -> None:
        tracks = [[7, None]]

        def provider(session: int, observation: int, target_session: int):
            del session, observation, target_session
            return []

        invalid_limits = (0, -1, True, 1.5, np.nan, np.inf, np.array([1]))
        for invalid_limit in invalid_limits:
            with self.subTest(max_path_length=invalid_limit):
                with self.assertRaisesRegex(ValueError, "max_path_length"):
                    enumerate_fragment_completion_paths(
                        tracks,
                        max_path_length=invalid_limit,
                        direction="suffix",
                        candidate_provider=provider,
                    )

    def test_rejects_invalid_max_paths_per_fragment_scalars(self) -> None:
        tracks = [[7, None]]

        def provider(session: int, observation: int, target_session: int):
            del session, observation, target_session
            return []

        invalid_limits = (0, -1, True, 1.5, np.nan, np.inf, np.array([1]))
        for invalid_limit in invalid_limits:
            with self.subTest(max_paths_per_fragment=invalid_limit):
                with self.assertRaisesRegex(ValueError, "max_paths_per_fragment"):
                    enumerate_fragment_completion_paths(
                        tracks,
                        direction="suffix",
                        candidate_provider=provider,
                        max_paths_per_fragment=invalid_limit,
                    )


if __name__ == "__main__":
    unittest.main()

