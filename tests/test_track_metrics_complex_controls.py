"""Regression tests for complex-valued track-latency controls."""

import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.utils.track_metrics import track_latencies


class TestTrackMetricsComplexControls(unittest.TestCase):
    def test_track_latency_rejects_complex_session_times(self):
        invalid_session_times = (
            np.array([0.0 + 0.0j, 1.0 + 0.0j]),
            np.array([0.0, np.complex128(1.0 + 2.0j)], dtype=object),
            [0.0, complex(1.0, 2.0)],
        )

        for session_times in invalid_session_times:
            with self.subTest(session_times=repr(session_times)):
                with self.assertRaisesRegex(
                    ValueError,
                    "session_times must contain only finite numeric values",
                ):
                    track_latencies(
                        [[None, 0]],
                        [[0, 0]],
                        session_times=session_times,
                    )

    def test_track_latency_rejects_complex_missed_values(self):
        invalid_missed_values = (
            complex(1.0, 0.0),
            np.complex64(1.0 + 2.0j),
            np.asarray(np.complex128(3.0 + 0.0j)),
            np.array(np.complex64(2.0 + 1.0j), dtype=object),
        )

        for missed_value in invalid_missed_values:
            with self.subTest(missed_value=repr(missed_value)):
                with self.assertRaisesRegex(ValueError, "missed_value"):
                    track_latencies([[None]], [[0]], missed_value=missed_value)

    def test_track_latency_preserves_real_numpy_scalar_controls(self):
        latencies = track_latencies(
            [[None, 0], [None, None]],
            [[0, 0], [1, 1]],
            session_times=np.array([np.float32(0.0), np.float64(2.0)]),
            missed_value=np.float32(-1.0),
        )

        npt.assert_allclose(latencies, np.array([2.0, -1.0]))


if __name__ == "__main__":
    unittest.main()
