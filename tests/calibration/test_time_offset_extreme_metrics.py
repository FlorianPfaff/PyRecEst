import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.calibration.time_offset import (
    aggregate_time_offset_sweeps,
    time_offset_error_summary,
)


class TimeOffsetExtremeMetricStabilityTest(unittest.TestCase):
    def test_summary_preserves_extreme_finite_residuals(self):
        expected = np.hypot(1e308, 1e308)

        with np.errstate(over="raise", invalid="raise"):
            summary = time_offset_error_summary(
                np.array([0.0, 1.0]),
                np.array([[1e308, 1e308], [1e308, 1e308]]),
                np.array([0.0, 1.0]),
                np.zeros((2, 2)),
                0.0,
            )

        self.assertEqual(summary["count"], 2.0)
        self.assertEqual(summary["coverage"], 1.0)
        npt.assert_allclose(summary["mean"], expected, rtol=1e-15)
        npt.assert_allclose(summary["std"], 0.0, atol=0.0)
        npt.assert_allclose(summary["rmse"], expected, rtol=1e-15)
        npt.assert_allclose(summary["p95"], expected, rtol=1e-15)
        npt.assert_allclose(summary["max"], expected, rtol=1e-15)

    def test_aggregation_preserves_extreme_finite_metrics(self):
        summary = {
            "time_offset_s": 0.0,
            "count": 2.0,
            "mean": 1e308,
            "std": 0.0,
            "rmse": 1e308,
            "p95": 1e308,
            "max": 1e308,
        }

        with np.errstate(over="raise", invalid="raise"):
            aggregated = aggregate_time_offset_sweeps([[summary]])

        self.assertEqual(len(aggregated), 1)
        self.assertEqual(aggregated[0]["count"], 2.0)
        npt.assert_allclose(aggregated[0]["mean"], 1e308, rtol=1e-15)
        npt.assert_allclose(aggregated[0]["std"], 0.0, atol=0.0)
        npt.assert_allclose(aggregated[0]["rmse"], 1e308, rtol=1e-15)
        npt.assert_allclose(aggregated[0]["p95"], 1e308, rtol=1e-15)
        npt.assert_allclose(aggregated[0]["max"], 1e308, rtol=1e-15)


if __name__ == "__main__":
    unittest.main()
