import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.calibration.time_offset import fit_time_offset


class TimeOffsetMetricValidationTest(unittest.TestCase):
    def _fit_with_metric(self, metric):
        reference_times = np.array([0.0, 1.0, 2.0])
        reference_values = reference_times.reshape(-1, 1)
        return fit_time_offset(
            reference_times,
            reference_values,
            reference_times,
            reference_values,
            np.array([0.0]),
            metric=metric,
        )

    def test_fit_time_offset_rejects_unknown_metric(self):
        with self.assertRaisesRegex(ValueError, "metric must be one of"):
            self._fit_with_metric("median")

    def test_fit_time_offset_rejects_non_string_metric(self):
        with self.assertRaisesRegex(ValueError, "metric must be one of"):
            self._fit_with_metric(["rmse"])

    def test_fit_time_offset_normalizes_metric_name(self):
        result = self._fit_with_metric("RMSE")

        self.assertEqual(result.metric, "rmse")
        self.assertEqual(result.best_offset_s, 0.0)
        npt.assert_allclose(result.metric_values, np.array([0.0]))


if __name__ == "__main__":
    unittest.main()
