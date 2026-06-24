import unittest

import numpy as np
from pyrecest.tracking.event_records import TrackingEvent, TrackingRecord


class TestTrackingEventScalarValidation(unittest.TestCase):
    def test_time_accepts_scalar_numeric_values(self):
        event = TrackingEvent(time=np.array(1.25), source="sensor")

        self.assertEqual(event.time, 1.25)

    def test_time_rejects_bool_and_non_scalar_values(self):
        for time in (True, np.bool_(False), np.array([1.0]), np.array([[1.0]])):
            with self.subTest(time=time):
                with self.assertRaisesRegex(ValueError, "time must be finite"):
                    TrackingEvent(time=time, source="sensor")


class TestTrackingRecordScalarValidation(unittest.TestCase):
    def _valid_record_kwargs(self):
        return {
            "time": 0.0,
            "source": "sensor",
            "action": "update",
            "prior_mean": np.array([0.0, 0.0]),
            "prior_cov": np.eye(2),
            "posterior_mean": np.array([1.0, 0.0]),
            "posterior_cov": np.eye(2),
        }

    def test_nis_accepts_nonnegative_scalar_numeric_values(self):
        record = TrackingRecord(**self._valid_record_kwargs(), nis=np.array(2.5))

        self.assertEqual(record.nis, 2.5)

    def test_time_rejects_bool_and_non_scalar_values(self):
        for time in (True, np.bool_(False), np.array([1.0]), np.array([[1.0]])):
            kwargs = self._valid_record_kwargs()
            kwargs["time"] = time
            with self.subTest(time=time):
                with self.assertRaisesRegex(ValueError, "time must be finite"):
                    TrackingRecord(**kwargs)

    def test_nis_rejects_bool_non_scalar_and_negative_values(self):
        invalid_nis_values = (True, np.bool_(False), np.array([1.0]), -1.0, np.nan)

        for nis in invalid_nis_values:
            with self.subTest(nis=nis):
                with self.assertRaisesRegex(
                    ValueError,
                    "nis must be finite and nonnegative",
                ):
                    TrackingRecord(**self._valid_record_kwargs(), nis=nis)


if __name__ == "__main__":
    unittest.main()
