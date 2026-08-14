import math
import unittest

import numpy as np
from pyrecest.evaluation.diagnostic_summaries import (
    covariance_inflation_summary,
    worst_time_windows,
)


class DiagnosticSummaryExtremeStatisticsTest(unittest.TestCase):
    def test_representable_large_statistics_remain_finite(self):
        large = 1.0e308
        records = [
            {
                "time_s": 0.0,
                "track_id": "A",
                "source": "radar",
                "error": large,
                "residual_norm": large,
                "covariance_scale": large,
            },
            {
                "time_s": 1.0,
                "track_id": "A",
                "source": "radar",
                "error": large,
                "residual_norm": large,
                "covariance_scale": large,
            },
        ]

        with np.errstate(all="raise"):
            window = worst_time_windows(records, window_s=5.0)[0]
            inflation = covariance_inflation_summary(records)

        for key in ("rmse", "mae", "p95", "mean_residual"):
            self.assertTrue(math.isfinite(window[key]))
            self.assertEqual(window[key], large)
        for key in ("mean_scale", "p95_scale"):
            self.assertTrue(math.isfinite(inflation[key]))
            self.assertEqual(inflation[key], large)


if __name__ == "__main__":
    unittest.main()
