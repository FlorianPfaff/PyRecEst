"""Regression tests for strict track-evaluation session selector validation."""

import unittest

import numpy as np
from pyrecest.utils.track_evaluation import (
    complete_track_set,
    score_complete_tracks,
    score_pairwise_tracks,
    score_track_matrices,
    track_error_ledger,
    track_pair_set,
)


class TestTrackEvaluationSessionValidation(unittest.TestCase):
    def test_complete_session_indices_reject_noninteger_values(self):
        invalid_indices = (
            True,
            np.bool_(True),
            1.5,
            np.nan,
            np.inf,
            "1",
            b"1",
            1 + 0j,
            np.array([1]),
        )

        for session_index in invalid_indices:
            with self.subTest(function="complete_track_set", value=session_index):
                with self.assertRaisesRegex(
                    ValueError,
                    "session_indices entries must be integer session indices",
                ):
                    complete_track_set([[0, 1]], session_indices=[session_index])
            with self.subTest(function="score_complete_tracks", value=session_index):
                with self.assertRaisesRegex(
                    ValueError,
                    "session_indices entries must be integer session indices",
                ):
                    score_complete_tracks(
                        [[0, 1]],
                        [[0, 1]],
                        session_indices=[session_index],
                    )

    def test_session_pairs_reject_noninteger_values(self):
        invalid_pairs = (
            (0, True),
            (0, np.bool_(True)),
            (0, 1.5),
            (0, np.nan),
            (0, np.inf),
            (0, "1"),
            (0, b"1"),
            (0, 1 + 0j),
            (0, np.array([1])),
        )

        for pair in invalid_pairs:
            with self.subTest(function="track_pair_set", pair=pair):
                with self.assertRaisesRegex(
                    ValueError,
                    "session_pairs entries must be integer session indices",
                ):
                    track_pair_set([[0, 1]], session_pairs=[pair])
            with self.subTest(function="score_pairwise_tracks", pair=pair):
                with self.assertRaisesRegex(
                    ValueError,
                    "session_pairs entries must be integer session indices",
                ):
                    score_pairwise_tracks([[0, 1]], [[0, 1]], session_pairs=[pair])
            with self.subTest(function="score_track_matrices", pair=pair):
                with self.assertRaisesRegex(
                    ValueError,
                    "session_pairs entries must be integer session indices",
                ):
                    score_track_matrices([[0, 1]], [[0, 1]], session_pairs=[pair])

    def test_session_pairs_reject_wrong_arity(self):
        invalid_pairs = ((0,), (0, 1, 2), 0)

        for pair in invalid_pairs:
            with self.subTest(pair=pair):
                with self.assertRaisesRegex(
                    ValueError,
                    "session_pairs must contain pairs of session indices",
                ):
                    track_pair_set([[0, 1]], session_pairs=[pair])
                with self.assertRaisesRegex(
                    ValueError,
                    "session_pairs must contain pairs of session indices",
                ):
                    track_error_ledger([[0, 1]], [[0, 1]], session_pairs=[pair])

    def test_integer_like_float_session_selectors_are_still_accepted(self):
        self.assertEqual(
            complete_track_set([[0, 1, 2]], session_indices=[0.0, np.float64(2.0)]),
            {(0, 2)},
        )
        self.assertEqual(
            track_pair_set([[0, 1, 2]], session_pairs=[(0.0, np.float64(2.0))]),
            {(0, 2, 0, 2)},
        )


if __name__ == "__main__":
    unittest.main()
