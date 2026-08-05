"""Regression tests for masked track-completion inputs."""

from __future__ import annotations

import unittest

import numpy as np
from pyrecest.utils.track_completion import (
    CompletionCandidate,
    enumerate_fragment_completion_paths,
)


class TestTrackCompletionMaskedValidation(unittest.TestCase):
    tracks = [[0, None]]

    @staticmethod
    def provider(session: int, observation: int, target_session: int):
        del session, observation, target_session
        return [1]

    def test_rejects_masked_public_controls(self) -> None:
        invalid_calls = (
            (
                "max_path_length",
                {"max_path_length": np.ma.array(2, mask=True)},
                "positive integer",
            ),
            (
                "max_paths_per_fragment",
                {"max_paths_per_fragment": np.ma.array(1, mask=True)},
                "positive integer",
            ),
            (
                "allow_duplicate_source",
                {"allow_duplicate_source": np.ma.array(True, mask=True)},
                "must be a boolean",
            ),
            (
                "allow_duplicate_target",
                {"allow_duplicate_target": np.ma.array(True, mask=True)},
                "must be a boolean",
            ),
        )
        for label, kwargs, message in invalid_calls:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, message):
                    enumerate_fragment_completion_paths(
                        self.tracks,
                        direction="suffix",
                        candidate_provider=self.provider,
                        **kwargs,
                    )

    def test_rejects_masked_candidate_observations_and_scores(self) -> None:
        invalid_candidates = (
            np.ma.array(1, mask=True),
            CompletionCandidate(np.ma.array(1, mask=True)),
        )
        for candidate in invalid_candidates:
            with self.subTest(candidate=repr(candidate)):
                with self.assertRaisesRegex(ValueError, "candidate observations"):
                    enumerate_fragment_completion_paths(
                        self.tracks,
                        direction="suffix",
                        candidate_provider=lambda *_args, value=candidate: [value],
                    )

        with self.assertRaisesRegex(ValueError, "candidate scores"):
            enumerate_fragment_completion_paths(
                self.tracks,
                direction="suffix",
                candidate_provider=lambda *_args: [
                    CompletionCandidate(1, score=np.ma.array(0.25, mask=True))
                ],
            )

        with self.assertRaisesRegex(ValueError, "path scores"):
            enumerate_fragment_completion_paths(
                self.tracks,
                direction="suffix",
                candidate_provider=self.provider,
                score_path=lambda _steps: np.ma.array(0.5, mask=True),
            )

    def test_ignores_masked_candidate_sessions(self) -> None:
        paths = enumerate_fragment_completion_paths(
            [[7, None, None]],
            direction="suffix",
            candidate_provider=lambda *_args: [9],
            candidate_session_provider=lambda *_args: [np.ma.array(2, mask=True)],
        )
        self.assertEqual(paths, [])

    def test_accepts_fully_unmasked_masked_scalar_wrappers(self) -> None:
        paths = enumerate_fragment_completion_paths(
            self.tracks,
            max_path_length=np.ma.array(1, mask=False),
            direction="suffix",
            candidate_provider=lambda *_args: [
                CompletionCandidate(
                    np.ma.array(1, mask=False),
                    score=np.ma.array(0.25, mask=False),
                )
            ],
            candidate_session_provider=lambda *_args: [np.ma.array(1, mask=False)],
            allow_duplicate_source=np.ma.array(False, mask=False),
            allow_duplicate_target=np.ma.array(False, mask=False),
            max_paths_per_fragment=np.ma.array(1, mask=False),
        )
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].end_observation, 1)
        self.assertEqual(paths[0].score, 0.25)


if __name__ == "__main__":
    unittest.main()
