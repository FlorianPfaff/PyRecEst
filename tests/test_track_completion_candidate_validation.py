"""Regression tests for fragment-completion candidate validation."""

from __future__ import annotations

import unittest

import numpy as np

from pyrecest.utils.track_completion import (
    CompletionCandidate,
    enumerate_fragment_completion_paths,
    path_observations,
)


class TestTrackCompletionCandidateValidation(unittest.TestCase):
    def test_numpy_scalar_array_candidates_do_not_truncate(self) -> None:
        tracks = [[0, None]]
        invalid_candidates = (
            np.array(1.5),
            np.array([1.5]),
            np.array(True),
            CompletionCandidate(np.array(1.5)),
        )

        for invalid_candidate in invalid_candidates:
            with self.subTest(invalid_candidate=repr(invalid_candidate)):

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

    def test_numpy_integer_scalar_array_candidates_are_accepted(self) -> None:
        tracks = [[0, None]]

        def provider(session: int, observation: int, target_session: int):
            del session, observation, target_session
            return [np.array(1.0)]

        paths = enumerate_fragment_completion_paths(
            tracks,
            direction="suffix",
            candidate_provider=provider,
        )

        self.assertEqual(len(paths), 1)
        self.assertEqual(path_observations(paths[0]), (0, 1))


if __name__ == "__main__":
    unittest.main()
