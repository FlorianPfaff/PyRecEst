"""Regression tests for masked track-metric controls."""

import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.utils.track_metrics import (
    score_false_tracks,
    score_missed_tracks,
    track_latencies,
)


class TestTrackMetricsMaskedInputs(unittest.TestCase):
    def setUp(self):
        self.reference = [[0, 0, None]]
        self.predicted = [[None, 0, None]]

    def test_masked_session_time_is_rejected(self):
        session_times = np.ma.array([0.0, 100.0, 200.0], mask=[False, True, False])

        with self.assertRaisesRegex(
            ValueError, "session_times must contain only finite numeric values"
        ):
            track_latencies(
                self.predicted,
                self.reference,
                session_times=session_times,
            )

    def test_masked_missed_value_is_rejected(self):
        missed_value = np.ma.array(-7.0, mask=True)

        with self.assertRaisesRegex(
            ValueError, "missed_value must be a scalar numeric value"
        ):
            track_latencies(
                [[None, None, None]],
                self.reference,
                missed_value=missed_value,
            )

    def test_masked_min_length_is_rejected(self):
        min_length = np.ma.array(2, mask=True)

        for scorer in (score_false_tracks, score_missed_tracks):
            with self.subTest(scorer=scorer.__name__):
                with self.assertRaisesRegex(
                    ValueError, "min_length must be a positive integer"
                ):
                    scorer(
                        [[99, 99, None]],
                        self.reference,
                        min_length=min_length,
                    )

    def test_unmasked_masked_arrays_remain_supported(self):
        session_times = np.ma.array([0.0, 4.0, 9.0], mask=False)
        missed_value = np.ma.array(-7.0, mask=False)
        min_length = np.ma.array(1, mask=False)

        npt.assert_allclose(
            track_latencies(
                self.predicted,
                self.reference,
                session_times=session_times,
            ),
            np.array([4.0]),
        )
        npt.assert_allclose(
            track_latencies(
                [[None, None, None]],
                self.reference,
                missed_value=missed_value,
            ),
            np.array([-7.0]),
        )
        self.assertEqual(
            score_false_tracks(
                [[99, None, None]],
                self.reference,
                min_length=min_length,
            )["false_track_evaluated_tracks"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
