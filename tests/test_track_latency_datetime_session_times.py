"""Regression tests for non-numeric track-latency session times."""

import unittest

import numpy as np
from pyrecest.utils.track_metrics import score_track_latency, track_latencies


class TrackLatencyDatetimeSessionTimesTest(unittest.TestCase):
    def setUp(self):
        self.predicted = [[0, 0, 0]]
        self.reference = [[0, 0, 0]]

    def test_numpy_datetime_like_session_times_are_rejected(self):
        invalid_session_times = (
            [
                np.datetime64("2026-01-01"),
                np.datetime64("2026-01-02"),
                np.datetime64("2026-01-03"),
            ],
            np.array(
                [
                    np.datetime64("2026-01-01"),
                    np.datetime64("2026-01-02"),
                    np.datetime64("2026-01-03"),
                ],
                dtype=object,
            ),
            [
                np.timedelta64(0, "s"),
                np.timedelta64(1, "s"),
                np.timedelta64(2, "s"),
            ],
            np.array(
                [
                    np.timedelta64(0, "s"),
                    np.timedelta64(1, "s"),
                    np.timedelta64(2, "s"),
                ],
                dtype=object,
            ),
        )

        for session_times in invalid_session_times:
            with self.subTest(session_times=repr(session_times)):
                with self.assertRaisesRegex(
                    ValueError,
                    "session_times must contain only finite numeric values",
                ):
                    track_latencies(
                        self.predicted,
                        self.reference,
                        session_times=session_times,
                    )
                with self.assertRaisesRegex(
                    ValueError,
                    "session_times must contain only finite numeric values",
                ):
                    score_track_latency(
                        self.predicted,
                        self.reference,
                        session_times=session_times,
                    )


if __name__ == "__main__":
    unittest.main()
