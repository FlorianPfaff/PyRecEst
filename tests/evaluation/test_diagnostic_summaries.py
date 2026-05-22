import math
import unittest

from pyrecest.evaluation.diagnostic_summaries import (
    build_diagnostic_summary,
    covariance_inflation_summary,
    top_residuals,
    track_switch_summary,
    worst_time_windows,
)


class DiagnosticSummariesTest(unittest.TestCase):
    def setUp(self):
        self.records = [
            {
                "time_s": 0.0,
                "track_id": "A",
                "source": "rf",
                "residual_norm": 1.0,
                "covariance_scale": 1.0,
                "error": 1.0,
            },
            {
                "time_s": 1.0,
                "track_id": "A",
                "source": "rf",
                "residual_norm": 4.0,
                "covariance_scale": 2.0,
                "error": 2.0,
            },
            {
                "time_s": 2.0,
                "track_id": "B",
                "source": "radar",
                "residual_norm": 3.0,
                "covariance_scale": 3.0,
                "error": 10.0,
            },
            {
                "time_s": 7.0,
                "track_id": "B",
                "source": "radar",
                "residual_norm": 2.0,
                "covariance_scale": 1.0,
                "error": 3.0,
            },
        ]

    def test_top_residuals_returns_largest_first(self):
        rows = top_residuals(self.records, top_n=2)

        self.assertEqual([row["residual_norm"] for row in rows], [4.0, 3.0])

    def test_track_switch_summary_counts_transitions(self):
        summary = track_switch_summary(self.records)

        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["top_transitions"][0]["from_track_id"], "A")
        self.assertEqual(summary["top_transitions"][0]["to_track_id"], "B")

    def test_covariance_inflation_summary_counts_by_source(self):
        summary = covariance_inflation_summary(self.records)

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["by_source"], {"radar": 1, "rf": 1})
        self.assertEqual(summary["max_scale"], 3.0)

    def test_worst_windows_sort_by_rmse(self):
        rows = worst_time_windows(self.records, window_s=5.0, top_n=2)

        self.assertEqual(rows[0]["time_start_s"], 0.0)
        self.assertTrue(math.isclose(rows[0]["rmse"], math.sqrt((1 + 4 + 100) / 3)))
        self.assertEqual(rows[0]["track_switch_count"], 1)

    def test_build_diagnostic_summary(self):
        summary = build_diagnostic_summary(self.records, top_n=2, window_s=5.0)

        self.assertEqual(summary["schema_version"], 1)
        self.assertEqual(len(summary["top_residuals"]), 2)
        self.assertEqual(summary["covariance_inflation"]["count"], 2)


if __name__ == "__main__":
    unittest.main()
