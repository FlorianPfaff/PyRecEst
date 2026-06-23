"""Tests for track-level outcome metrics."""

import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.utils.track_metrics import (
    false_track_rate,
    missed_track_rate,
    score_false_tracks,
    score_missed_tracks,
    score_track_latency,
    score_track_outcomes,
    score_track_purity,
    track_latencies,
    track_purity,
)


class TestTrackMetrics(unittest.TestCase):
    def setUp(self):
        self.reference = [
            [0, 0, 0],
            [1, 1, 1],
        ]
        self.predicted = [
            [0, 0, None],
            [None, None, 0],
            [1, 1, 1],
            [2, None, None],
        ]

    def test_track_purity_and_detection(self):
        purity = score_track_purity(self.predicted, self.reference)
        false_scores = score_false_tracks(self.predicted, self.reference)
        missed_scores = score_missed_tracks(self.predicted, self.reference)

        npt.assert_allclose(purity["observation_weighted_track_purity"], 6.0 / 7.0)
        npt.assert_allclose(track_purity(self.predicted, self.reference), 6.0 / 7.0)
        self.assertEqual(purity["pure_tracks"], 3)
        self.assertEqual(purity["impure_tracks"], 1)
        self.assertEqual(false_scores["false_tracks"], 1)
        self.assertEqual(missed_scores["missed_tracks"], 0)
        npt.assert_allclose(false_track_rate(self.predicted, self.reference), 0.25)
        npt.assert_allclose(missed_track_rate(self.predicted, self.reference), 0.0)

    def test_track_latency_and_combined_scores(self):
        npt.assert_allclose(
            track_latencies(self.predicted, self.reference), np.array([0.0, 0.0])
        )
        summary = score_track_latency(
            self.predicted, self.reference, session_times=[0.0, 5.0, 9.0]
        )
        self.assertEqual(summary["latency_detected_tracks"], 2)
        self.assertEqual(summary["mean_track_latency"], 0.0)

        delayed = score_track_latency([[None, 1, 1]], [[0, 1, 1]])
        self.assertEqual(delayed["mean_track_latency"], 1.0)

        scores = score_track_outcomes(self.predicted, self.reference)
        self.assertEqual(scores["fragmentation_fragmented_reference_tracks"], 1)
        self.assertEqual(scores["false_tracks"], 1)
        npt.assert_allclose(scores["observation_weighted_track_purity"], 6.0 / 7.0)

    def test_track_latency_rejects_nonfinite_session_times(self):
        invalid_session_times = (
            [0.0, np.nan, 2.0],
            [0.0, np.inf, 2.0],
            [0.0, -np.inf, 2.0],
        )

        for session_times in invalid_session_times:
            with self.subTest(session_times=session_times):
                with self.assertRaisesRegex(
                    ValueError,
                    "session_times must contain only finite values",
                ):
                    track_latencies(
                        self.predicted,
                        self.reference,
                        session_times=session_times,
                    )
                with self.assertRaisesRegex(
                    ValueError,
                    "session_times must contain only finite values",
                ):
                    score_track_latency(
                        self.predicted,
                        self.reference,
                        session_times=session_times,
                    )

    def test_min_length_limits_false_track_observation_rate_denominator(self):
        predicted = [
            [99, None, None],
            [10, 11, None],
        ]
        reference = [[10, 10, None]]

        scores = score_false_tracks(predicted, reference, min_length=2)

        self.assertEqual(scores["false_track_evaluated_tracks"], 1)
        self.assertEqual(scores["unreferenced_predicted_observations"], 1)
        npt.assert_allclose(scores["unreferenced_predicted_observation_rate"], 0.5)

    def test_min_length_limits_missed_track_observation_rate_denominator(self):
        predicted = [[10, 10, None]]
        reference = [
            [99, None, None],
            [10, 10, 10],
        ]

        scores = score_missed_tracks(predicted, reference, min_length=2)

        self.assertEqual(scores["missed_track_evaluated_reference_tracks"], 1)
        self.assertEqual(scores["missed_reference_observations"], 1)
        npt.assert_allclose(scores["missed_reference_observation_rate"], 1.0 / 3.0)

    def test_min_length_rejects_noninteger_values(self):
        invalid_values = (0, -1, 1.5, np.nan, np.inf, True, np.array([1]))

        for min_length in invalid_values:
            with self.subTest(function="score_false_tracks", min_length=min_length):
                with self.assertRaisesRegex(
                    ValueError,
                    "min_length must be a positive integer",
                ):
                    score_false_tracks(
                        self.predicted,
                        self.reference,
                        min_length=min_length,
                    )
            with self.subTest(function="score_missed_tracks", min_length=min_length):
                with self.assertRaisesRegex(
                    ValueError,
                    "min_length must be a positive integer",
                ):
                    score_missed_tracks(
                        self.predicted,
                        self.reference,
                        min_length=min_length,
                    )

    def test_min_length_accepts_integer_like_scalars(self):
        false_scores = score_false_tracks(
            self.predicted,
            self.reference,
            min_length=np.array(2.0),
        )
        missed_scores = score_missed_tracks(
            self.predicted,
            self.reference,
            min_length=2.0,
        )

        self.assertEqual(false_scores["false_track_evaluated_tracks"], 2)
        self.assertEqual(missed_scores["missed_track_evaluated_reference_tracks"], 2)


if __name__ == "__main__":
    unittest.main()
