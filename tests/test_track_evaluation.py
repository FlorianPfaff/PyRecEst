"""Tests for generic multi-session track-matrix evaluation helpers."""

import unittest

import numpy as np

from pyrecest.utils.track_evaluation import (
    complete_track_set,
    normalize_track_matrix,
    reference_fragment_counts,
    score_complete_tracks,
    score_false_continuations,
    score_pairwise_tracks,
    score_track_fragmentation,
    score_track_links,
    score_track_matrices,
    summarize_track_errors,
    summarize_tracks,
    track_error_ledger,
    track_lengths,
    track_pair_set,
)


class TestTrackEvaluation(unittest.TestCase):
    def test_normalize_track_matrix_parses_common_missing_values(self):
        matrix = normalize_track_matrix([[0, "1", None, -1, np.nan, "", b"2", 3.0, 4.5, "None", "null"]])

        self.assertEqual(matrix.shape, (1, 11))
        self.assertEqual(matrix[0, 0], 0)
        self.assertEqual(matrix[0, 1], 1)
        self.assertIsNone(matrix[0, 2])
        self.assertIsNone(matrix[0, 3])
        self.assertIsNone(matrix[0, 4])
        self.assertIsNone(matrix[0, 5])
        self.assertEqual(matrix[0, 6], 2)
        self.assertEqual(matrix[0, 7], 3)
        self.assertIsNone(matrix[0, 8])
        self.assertIsNone(matrix[0, 9])
        self.assertIsNone(matrix[0, 10])

    def test_track_lengths_and_complete_tracks(self):
        tracks = [[0, 1, 2], [3, None, 4], [None, 5, 6]]

        np.testing.assert_array_equal(track_lengths(tracks), np.asarray([3, 2, 2]))
        self.assertEqual(complete_track_set(tracks), {(0, 1, 2)})
        self.assertEqual(complete_track_set(tracks, session_indices=[1, 2]), {(1, 2), (5, 6)})
        self.assertEqual(summarize_tracks(tracks), {"tracks": 3, "mean_track_length": 7.0 / 3.0, "max_track_length": 3})

    def test_track_pair_set_defaults_to_adjacent_sessions(self):
        tracks = [[0, 1, 2], [3, None, 4]]

        self.assertEqual(track_pair_set(tracks), {(0, 1, 0, 1), (1, 2, 1, 2)})
        self.assertEqual(track_pair_set(tracks, session_pairs=[(0, 2)]), {(0, 2, 0, 2), (0, 2, 3, 4)})

    def test_score_track_links_and_pairwise_alias(self):
        predicted = [[0, 1, 2], [3, None, 5]]
        reference = [[0, 1, 2], [3, None, 4]]

        scores = score_track_links(predicted, reference)
        self.assertEqual(scores["track_link_true_positives"], 2)
        self.assertEqual(scores["track_link_false_positives"], 0)
        self.assertEqual(scores["track_link_false_negatives"], 0)
        self.assertEqual(scores["track_link_precision"], 1.0)

        pairwise_scores = score_pairwise_tracks(predicted, reference, session_pairs=[(0, 2)])
        self.assertEqual(pairwise_scores["pairwise_true_positives"], 1)
        self.assertEqual(pairwise_scores["pairwise_false_positives"], 1)
        self.assertEqual(pairwise_scores["pairwise_false_negatives"], 1)

    def test_score_complete_tracks(self):
        predicted = [[0, 1, 2], [3, None, 4]]
        reference = [[0, 1, 2], [5, 6, 7]]

        scores = score_complete_tracks(predicted, reference)
        self.assertEqual(scores["complete_track_true_positives"], 1)
        self.assertEqual(scores["complete_track_false_positives"], 0)
        self.assertEqual(scores["complete_track_false_negatives"], 1)
        self.assertAlmostEqual(scores["complete_track_recall"], 0.5)

    def test_score_false_continuations(self):
        predicted = [[0, 9], [7, 8], [42, 43]]
        reference = [[0, 1], [7, 8]]

        scores = score_false_continuations(predicted, reference)
        self.assertEqual(scores["valid_continuations"], 1)
        self.assertEqual(scores["false_continuations"], 1)
        self.assertEqual(scores["unknown_source_continuations"], 1)
        self.assertAlmostEqual(scores["false_continuation_rate"], 0.5)

    def test_fragmentation_counts_split_reference_tracks(self):
        predicted = [[0, 1, None], [None, None, 2], [3, 4, 5]]
        reference = [[0, 1, 2], [3, 4, 5]]

        np.testing.assert_array_equal(reference_fragment_counts(predicted, reference), np.asarray([2, 1]))
        scores = score_track_fragmentation(predicted, reference)
        self.assertEqual(scores["fragmentation_fragmented_reference_tracks"], 1)
        self.assertEqual(scores["fragmentation_events"], 1)
        self.assertEqual(scores["fragmentation_fragments"], 3)

    def test_track_error_ledger_reports_mixed_spurious_missed_and_duplicates(self):
        predicted = [[0, 6, 2], [None, 1, None], [8, None, None], [0, None, None]]
        reference = [[0, 1, 2], [5, 6, 7], [9, 10, 11]]

        ledger = track_error_ledger(predicted, reference)
        summary = ledger["summary"]
        self.assertEqual(summary["mixed_identity_tracks"], 1)
        self.assertEqual(summary["spurious_tracks"], 1)
        self.assertEqual(summary["missed_reference_tracks"], 1)
        self.assertEqual(summary["fragmented_reference_tracks"], 1)
        self.assertEqual(summary["predicted_duplicate_observations"], 1)
        self.assertTrue(ledger["link_errors"])
        self.assertEqual(ledger["duplicate_observations"][0]["observation"], 0)

    def test_summarize_track_errors_matches_ledger_summary(self):
        predicted = [[0, 10], [None, 1]]
        reference = [[0, 1]]

        self.assertEqual(summarize_track_errors(predicted, reference), track_error_ledger(predicted, reference)["summary"])

    def test_score_track_matrices_combines_metric_families(self):
        predicted = [[0, 1, None], [None, None, 2]]
        reference = [[0, 1, 2]]

        scores = score_track_matrices(predicted, reference)
        self.assertIn("track_link_f1", scores)
        self.assertIn("pairwise_f1", scores)
        self.assertIn("complete_track_f1", scores)
        self.assertIn("fragmentation_events", scores)
        self.assertIn("missed_reference_links", scores)
        self.assertIn("false_continuation_link_rate", scores)

    def test_rejects_invalid_shapes_and_session_pairs(self):
        with self.assertRaises(ValueError):
            normalize_track_matrix([0, 1, 2])
        with self.assertRaises(ValueError):
            score_track_links([[0, 1]], [[0, 1, 2]])
        with self.assertRaises(ValueError):
            track_pair_set([[0, 1]], session_pairs=[(1, 0)])
        with self.assertRaises(IndexError):
            track_pair_set([[0, 1]], session_pairs=[(0, 2)])


if __name__ == "__main__":
    unittest.main()
