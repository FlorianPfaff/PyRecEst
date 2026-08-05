import unittest

import numpy as np
import numpy.testing as npt

import pyrecest.calibration.time_offset as time_offset_module
from pyrecest.calibration import fit_time_offset, time_offset_error_summary


class TimeOffsetMetricStabilityTest(unittest.TestCase):
    def test_public_summary_preserves_large_finite_residuals(self):
        scale = 1.0e200

        with np.errstate(all="raise"):
            summary = time_offset_error_summary(
                np.array([0.0]),
                np.array([[scale, scale]]),
                np.array([0.0, 1.0]),
                np.array([[0.0, 0.0], [0.0, 0.0]]),
                0.0,
            )

        expected = np.sqrt(2.0) * scale
        self.assertEqual(summary["count"], 1.0)
        self.assertEqual(summary["coverage"], 1.0)
        self.assertEqual(summary["std"], 0.0)
        for key in ("mean", "rmse", "p95", "max"):
            self.assertTrue(np.isfinite(summary[key]))
            npt.assert_allclose(summary[key], expected, rtol=1.0e-15)

    def test_package_and_module_exports_share_stable_summary(self):
        self.assertIs(
            time_offset_error_summary,
            time_offset_module.time_offset_error_summary,
        )

    def test_fit_time_offset_keeps_large_finite_candidates(self):
        scale = 1.0e200

        with np.errstate(all="raise"):
            result = fit_time_offset(
                np.array([0.0]),
                np.array([[scale, scale]]),
                np.array([0.0, 1.0]),
                np.array(
                    [
                        [0.0, 0.0],
                        [0.5 * scale, 0.5 * scale],
                    ]
                ),
                np.array([0.0, 1.0]),
            )

        self.assertEqual(result.best_offset_s, 1.0)
        npt.assert_array_equal(result.counts, np.array([1, 1]))
        self.assertTrue(np.isfinite(result.metric_values).all())
        self.assertLess(result.metric_values[1], result.metric_values[0])


if __name__ == "__main__":
    unittest.main()
