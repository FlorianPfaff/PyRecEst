import unittest

import numpy as np
from pyrecest.filters.track_manager import TrackManager


class TrackManagerAtomicFilterStateTest(unittest.TestCase):
    @staticmethod
    def _state(mean):
        return np.array([mean], dtype=float), np.array([[1.0]], dtype=float)

    def test_invalid_replacement_preserves_existing_track_bank(self):
        manager = TrackManager(
            keep_history=False,
            log_prior_estimates=False,
            log_posterior_estimates=False,
        )
        manager.filter_state = [self._state(1.0)]
        original_tracks = manager.tracks
        original_track = manager.tracks[0]
        original_next_track_id = manager._next_track_id

        with self.assertRaisesRegex(ValueError, "Expected an AbstractFilter"):
            manager.filter_state = [self._state(2.0), object()]

        self.assertIs(manager.tracks, original_tracks)
        self.assertIs(manager.tracks[0], original_track)
        self.assertEqual(manager._next_track_id, original_next_track_id)
        np.testing.assert_allclose(
            manager.tracks[0].get_point_estimate(),
            np.array([1.0]),
        )
