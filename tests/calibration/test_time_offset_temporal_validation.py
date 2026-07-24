import unittest

import numpy as np
from pyrecest.calibration.time_offset import (
    aggregate_time_offset_sweeps,
    apply_time_offset,
    interpolate_reference_values,
)


class TimeOffsetTemporalValidationTest(unittest.TestCase):
    def test_apply_time_offset_rejects_temporal_offset_scalars(self):
        invalid_offsets = (
            np.timedelta64(2, "ns"),
            np.array(np.timedelta64(2, "ns"), dtype=object),
        )
        for offset_s in invalid_offsets:
            with self.subTest(offset_s=offset_s):
                with self.assertRaisesRegex(
                    ValueError,
                    "offset_s must be a finite scalar",
                ):
                    apply_time_offset(np.array([0.0]), offset_s)

    def test_apply_time_offset_rejects_object_wrapped_temporal_times(self):
        for times_s in (
            np.array([np.datetime64("2026-01-01")], dtype=object),
            np.array([np.timedelta64(2, "ns")], dtype=object),
        ):
            with self.subTest(times_s=times_s):
                with self.assertRaisesRegex(
                    ValueError,
                    "times_s must contain real numeric values",
                ):
                    apply_time_offset(times_s, 0.0)

    def test_interpolation_rejects_temporal_max_time_delta_scalars(self):
        invalid_deltas = (
            np.timedelta64(2, "ns"),
            np.array(np.timedelta64(2, "ns"), dtype=object),
        )
        for max_time_delta_s in invalid_deltas:
            with self.subTest(max_time_delta_s=max_time_delta_s):
                with self.assertRaisesRegex(
                    ValueError,
                    "max_time_delta_s must be nonnegative",
                ):
                    interpolate_reference_values(
                        np.array([0.0, 1.0]),
                        np.array([0.0, 1.0]),
                        np.array([0.5]),
                        max_time_delta_s=max_time_delta_s,
                    )

    def test_interpolation_rejects_object_wrapped_temporal_reference_values(self):
        reference_values = np.array(
            [np.timedelta64(1, "ns"), np.timedelta64(2, "ns")],
            dtype=object,
        )

        with self.assertRaisesRegex(
            ValueError,
            "reference_values must contain real numeric values",
        ):
            interpolate_reference_values(
                np.array([0.0, 1.0]),
                reference_values,
                np.array([0.5]),
            )

    def test_aggregate_sweeps_rejects_temporal_object_summary_scalars(self):
        sweep = [
            {
                "time_offset_s": np.array(np.timedelta64(2, "ns"), dtype=object),
                "count": 1.0,
                "mean": 1.0,
                "rmse": 1.0,
                "p95": 1.0,
                "max": 1.0,
            }
        ]

        with self.assertRaisesRegex(
            ValueError,
            "time_offset_s must be a real scalar",
        ):
            aggregate_time_offset_sweeps([sweep])


if __name__ == "__main__":
    unittest.main()
