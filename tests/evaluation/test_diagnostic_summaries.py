import math
import unittest

import numpy as np
from pyrecest.evaluation.diagnostic_summaries import (
    build_diagnostic_summary,
    covariance_inflation_summary,
    top_residuals,
    track_switch_summary,
    worst_time_windows,
)


class _OverflowFloat:
    def __float__(self):
        raise OverflowError("simulated overflow")


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

    def test_summary_rejects_invalid_top_n(self):
        for top_n in (
            0,
            1.5,
            math.nan,
            math.inf,
            True,
            "2",
            b"2",
            np.array("2"),
            np.array([1]),
        ):
            with self.subTest(top_n=top_n):
                with self.assertRaisesRegex(ValueError, "top_n"):
                    build_diagnostic_summary(self.records, top_n=top_n)
                with self.assertRaisesRegex(ValueError, "top_n"):
                    top_residuals(self.records, top_n=top_n)
                with self.assertRaisesRegex(ValueError, "top_n"):
                    track_switch_summary(self.records, top_n=top_n)
                with self.assertRaisesRegex(ValueError, "top_n"):
                    covariance_inflation_summary(self.records, top_n=top_n)
                with self.assertRaisesRegex(ValueError, "top_n"):
                    worst_time_windows(self.records, top_n=top_n)

    def test_summary_rejects_invalid_window_s(self):
        for window_s in (
            0.0,
            -1.0,
            math.nan,
            math.inf,
            True,
            "5.0",
            b"5.0",
            np.array("5.0"),
            np.array([1.0]),
        ):
            with self.subTest(window_s=window_s):
                with self.assertRaisesRegex(ValueError, "window_s"):
                    build_diagnostic_summary(self.records, window_s=window_s)
                with self.assertRaisesRegex(ValueError, "window_s"):
                    worst_time_windows(self.records, window_s=window_s)

    def test_summary_normalizes_numeric_scalar_parameters(self):
        summary = build_diagnostic_summary(
            self.records,
            top_n=np.array(2.0),
            window_s=np.array(5.0),
        )

        self.assertEqual(summary["top_n"], 2)
        self.assertEqual(summary["window_s"], 5.0)

    def test_summary_treats_overflowing_optional_scalars_as_missing(self):
        records = [
            {
                "time_s": _OverflowFloat(),
                "track_id": _OverflowFloat(),
                "source": "rf",
                "residual_norm": _OverflowFloat(),
                "covariance_scale": _OverflowFloat(),
                "error": _OverflowFloat(),
            }
        ]

        self.assertEqual(top_residuals(records), [])
        self.assertEqual(track_switch_summary(records)["updates_with_track_id"], 0)
        self.assertEqual(covariance_inflation_summary(records)["count"], 0)
        self.assertEqual(worst_time_windows(records), [])

        summary = build_diagnostic_summary(records, top_n=2, window_s=5.0)
        self.assertEqual(summary["top_residuals"], [])
        self.assertEqual(summary["track_switches"]["updates_with_track_id"], 0)
        self.assertEqual(summary["covariance_inflation"]["count"], 0)
        self.assertEqual(summary["worst_time_windows"], [])


if __name__ == "__main__":
    unittest.main()
