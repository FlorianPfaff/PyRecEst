import re
import unittest

import numpy as np
from pyrecest.utils import CandidatePruningConfig, prune_pairwise_cost_matrix


class TestCandidatePruningTemporalScalars(unittest.TestCase):
    def test_scalar_cost_controls_reject_temporal_scalars(self):
        temporal_values = (
            np.timedelta64(1, "ns"),
            np.datetime64("1970-01-01T00:00:00.000000001"),
            np.asarray(np.timedelta64(1, "ns")),
            np.asarray(np.datetime64("1970-01-01T00:00:00.000000001")),
        )
        cases = (
            (
                "probability_threshold",
                "probability_threshold must lie in [0, 1]",
            ),
            ("max_cost", "max_cost must be finite or None"),
            (
                "max_cost_percentile",
                "max_cost_percentile must lie in [0, 100]",
            ),
            ("large_cost", "large_cost must be finite and positive"),
        )

        for field_name, message in cases:
            for value in temporal_values:
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaisesRegex(ValueError, re.escape(message)):
                        CandidatePruningConfig(**{field_name: value})

        with self.assertRaisesRegex(
            ValueError,
            "large_cost must be finite and positive",
        ):
            prune_pairwise_cost_matrix(
                np.array([[1.0, 2.0]]),
                config=CandidatePruningConfig(row_top_k=1),
                large_cost=np.timedelta64(2, "ns"),
            )

    def test_top_k_rejects_temporal_scalars(self):
        temporal_top_k_values = (
            np.timedelta64(2, "ns"),
            np.datetime64("1970-01-01T00:00:00.000000002"),
            np.asarray(np.timedelta64(2, "ns")),
            np.asarray(np.datetime64("1970-01-01T00:00:00.000000002")),
        )

        for field_name in ("row_top_k", "column_top_k"):
            for value in temporal_top_k_values:
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaisesRegex(
                        ValueError,
                        f"{field_name} must be a positive integer or None",
                    ):
                        CandidatePruningConfig(**{field_name: value})


if __name__ == "__main__":
    unittest.main()
