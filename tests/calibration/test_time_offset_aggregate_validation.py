import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.calibration import aggregate_time_offset_sweeps
from pyrecest.calibration.time_offset import (
    aggregate_time_offset_sweeps as aggregate_time_offset_sweeps_from_module,
)


class TimeOffsetAggregateValidationTest(unittest.TestCase):
    def _summary_row(self) -> dict[str, float]:
        return {
            "time_offset_s": 0.0,
            "count": 2.0,
            "mean": 1.0,
            "std": 0.5,
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

    def test_aggregate_time_offset_sweeps_rejects_invalid_metric_names(self):
        invalid_metrics = (None, 1, "", "median", "coverage")
        aggregators = (
            aggregate_time_offset_sweeps,
            aggregate_time_offset_sweeps_from_module,
        )
        for aggregate in aggregators:
            for metric in invalid_metrics:
                with self.subTest(aggregate=aggregate.__module__, metric=metric):
                    with self.assertRaisesRegex(ValueError, "metric must be one of"):
                        aggregate([[self._summary_row()]], metric=metric)

    def test_aggregate_time_offset_sweeps_normalizes_metric_names(self):
        for aggregate in (
            aggregate_time_offset_sweeps,
            aggregate_time_offset_sweeps_from_module,
        ):
            with self.subTest(aggregate=aggregate.__module__):
                aggregated = aggregate(
                    [[self._summary_row()]],
                    metric=" STD ",
                )

                self.assertEqual(aggregated[0]["std"], 0.5)
                self.assertNotIn(" STD ", aggregated[0])


if __name__ == "__main__":
    unittest.main()
