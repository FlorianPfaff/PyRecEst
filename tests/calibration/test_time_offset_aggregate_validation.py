import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.calibration.time_offset import aggregate_time_offset_sweeps


class TimeOffsetAggregateValidationTest(unittest.TestCase):
    def _summary_row(self) -> dict[str, float]:
        return {
            "time_offset_s": 0.0,
            "count": 2.0,
            "mean": 1.0,
            "rmse": 2.0,
            "p95": 2.5,
            "max": 3.0,
        }

    def test_aggregate_time_offset_sweeps_accepts_scalar_numpy_values(self):
        row = self._summary_row()
        row.update(
            {
                "time_offset_s": np.array(0.0),
                "count": np.float64(2.0),
                "mean": np.array(1.0),
                "rmse": np.float64(2.0),
            }
        )

        aggregated = aggregate_time_offset_sweeps([[row]])

        self.assertEqual(aggregated[0]["count"], 2.0)
        npt.assert_allclose(aggregated[0]["rmse"], 2.0)

    def test_aggregate_time_offset_sweeps_rejects_malformed_summary_scalars(self):
        invalid_cases = (
            ("time_offset_s", "0.0", "time_offset_s"),
            ("count", True, "count"),
            ("count", -1.0, "count"),
            ("mean", 1.0 + 0.0j, "mean"),
            ("rmse", "2.0", "rmse"),
            ("p95", np.array([2.5]), "p95"),
        )
        for key, value, pattern in invalid_cases:
            with self.subTest(key=key, value=value):
                row = self._summary_row()
                row[key] = value
                with self.assertRaisesRegex(ValueError, pattern):
                    aggregate_time_offset_sweeps([[row]])


if __name__ == "__main__":
    unittest.main()
