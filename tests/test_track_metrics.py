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


if __name__ == "__main__":
    unittest.main()
