import unittest

import numpy as np
import numpy.testing as npt

from pyrecest.calibration.time_offset import (
    aggregate_time_offset_sweeps,
    time_offset_error_summary,
)


class TimeOffsetLargeFiniteErrorTest(unittest.TestCase):
    def test_summary_preserves_large_finite_errors(self):
        large = np.finfo(float).max * 0.75

        summary = time_offset_error_summary(
            np.array([0.25, 0.75]),
            np.array([[large], [large]]),
            np.array([0.0, 1.0]),
            np.array([[0.0], [0.0]]),
            0.0,
        )

        self.assertEqual(summary["count"], 2.0)
        self.assertEqual(summary["coverage"], 1.0)
        self.assertEqual(summary["std"], 0.0)
        for key in ("mean", "rmse", "p95", "max"):
            with self.subTest(key=key):
                self.assertTrue(np.isfinite(summary[key]))
                npt.assert_allclose(summary[key], large, rtol=1e-15)

    def test_aggregation_preserves_large_finite_metrics(self):
        large = np.finfo(float).max * 0.75
        part = {
            "time_offset_s": 0.0,
            "count": 1.0,
            "mean": large,
            "std": 0.0,
            "rmse": large,
            "p95": large,
            "max": large,
        }

        aggregated = aggregate_time_offset_sweeps([[part], [part]])[0]

        self.assertEqual(aggregated["count"], 2.0)
        self.assertEqual(aggregated["std"], 0.0)
        for key in ("mean", "rmse", "p95", "max"):
            with self.subTest(key=key):
                self.assertTrue(np.isfinite(aggregated[key]))
                npt.assert_allclose(aggregated[key], large, rtol=1e-15)


if __name__ == "__main__":
    unittest.main()
