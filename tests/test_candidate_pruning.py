import re
import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.utils import (
    CandidatePruningConfig,
    candidate_mask_from_costs,
    candidate_pruning_config_from_mapping,
    prune_pairwise_cost_matrix,
)


class TestCandidatePruning(unittest.TestCase):
    def test_none_config_keeps_all_finite_costs(self):
        costs = np.array([[1.0, np.inf], [np.nan, 2.0]])

        mask = candidate_mask_from_costs(costs)
        pruned = prune_pairwise_cost_matrix(costs)

        npt.assert_array_equal(mask, np.array([[True, False], [False, True]]))
        npt.assert_allclose(pruned, np.array([[1.0, np.inf], [np.inf, 2.0]]))

    def test_row_and_column_top_k_rules_are_unioned(self):
        costs = np.array(
            [
                [4.0, 1.0, 3.0],
                [2.0, 5.0, 0.5],
                [7.0, 0.2, 6.0],
            ]
        )

        mask = candidate_mask_from_costs(
            costs,
            config=CandidatePruningConfig(row_top_k=1, column_top_k=1),
        )

        expected = np.array(
            [
                [False, True, False],
                [True, False, True],
                [False, True, False],
            ]
        )
        npt.assert_array_equal(mask, expected)

    def test_probability_and_cost_threshold_rules_are_unioned(self):
        costs = np.array([[1.0, 3.0], [4.0, 0.8]])
        probabilities = np.array([[0.1, 0.95], [0.2, 0.3]])

        mask = candidate_mask_from_costs(
            costs,
            probability_matrix=probabilities,
            config={"probability_threshold": 0.9, "max_cost": 1.0},
        )

        npt.assert_array_equal(mask, np.array([[True, True], [False, True]]))

    def test_probability_threshold_requires_probability_matrix(self):
        costs = np.array([[1.0, 2.0]])
        config = CandidatePruningConfig(probability_threshold=0.9)

        with self.assertRaisesRegex(
            ValueError,
            "probability_matrix is required when probability_threshold is set",
        ):
            candidate_mask_from_costs(costs, config=config)

        with self.assertRaisesRegex(
            ValueError,
            "probability_matrix is required when probability_threshold is set",
        ):
            prune_pairwise_cost_matrix(costs, config=config)

    def test_probability_matrix_rejects_finite_values_outside_unit_interval(self):
        config = CandidatePruningConfig(probability_threshold=0.5)

        for probabilities in (np.array([[1.2, 0.5]]), np.array([[0.5, -0.1]])):
            with self.subTest(probabilities=probabilities):
                with self.assertRaisesRegex(
                    ValueError,
                    "finite probability_matrix entries must lie in \\[0, 1\\]",
                ):
                    candidate_mask_from_costs(
                        np.array([[1.0, 2.0]]),
                        probability_matrix=probabilities,
                        config=config,
                    )

    def test_probability_matrix_ignores_nonfinite_values(self):
        config = CandidatePruningConfig(probability_threshold=0.5)

        mask = candidate_mask_from_costs(
            np.array([[1.0, 2.0]]),
            probability_matrix=np.array([[np.nan, 0.75]]),
            config=config,
        )

        npt.assert_array_equal(mask, np.array([[False, True]]))

    def test_percentile_rule_and_large_cost_replacement(self):
        costs = np.array([[1.0, 2.0, 100.0], [3.0, 4.0, 5.0]])

        pruned = prune_pairwise_cost_matrix(
            costs,
            config=CandidatePruningConfig(max_cost_percentile=50.0, large_cost=999.0),
        )

        npt.assert_allclose(
            pruned,
            np.array([[1.0, 2.0, 999.0], [3.0, 999.0, 999.0]]),
        )

    def test_pruned_entries_remain_more_expensive_than_large_finite_costs(self):
        costs = np.array([[2.0e6, 3.0e6]])

        pruned = prune_pairwise_cost_matrix(
            costs,
            config=CandidatePruningConfig(row_top_k=1),
        )

        self.assertEqual(pruned[0, 0], costs[0, 0])
        self.assertTrue(np.isfinite(pruned[0, 1]))
        self.assertGreater(pruned[0, 1], costs[0, 1])

    def test_always_keep_finite_overrides_selective_rules(self):
        costs = np.array([[1.0, 10.0], [3.0, np.inf]])

        mask = candidate_mask_from_costs(
            costs,
            config=CandidatePruningConfig(
                row_top_k=1,
                always_keep_finite=True,
            ),
        )

        npt.assert_array_equal(mask, np.array([[True, True], [True, False]]))

    def test_mapping_config_normalization_and_validation(self):
        cfg = candidate_pruning_config_from_mapping({"row_top_k": np.array(2.0)})
        self.assertIsInstance(cfg, CandidatePruningConfig)
        self.assertEqual(cfg.row_top_k, 2)

        with self.assertRaises(ValueError):
            CandidatePruningConfig(row_top_k=0)
        with self.assertRaises(ValueError):
            CandidatePruningConfig(probability_threshold=1.1)
        with self.assertRaises(ValueError):
            candidate_mask_from_costs(np.array([1.0, 2.0]))
        with self.assertRaises(ValueError):
            candidate_mask_from_costs(
                np.array([[1.0]]),
                probability_matrix=np.array([[0.5, 0.5]]),
                config=CandidatePruningConfig(probability_threshold=0.4),
            )

    def test_scalar_cost_controls_reject_bools_and_non_scalars(self):
        invalid_values = (True, np.array([1.0]))
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
            for value in invalid_values:
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaisesRegex(ValueError, re.escape(message)):
                        CandidatePruningConfig(**{field_name: value})

        with self.assertRaisesRegex(
            ValueError,
            "large_cost must be finite and positive",
        ):
            prune_pairwise_cost_matrix(
                np.array([[1.0]]),
                config=CandidatePruningConfig(row_top_k=1),
                large_cost=True,
            )

    def test_top_k_rejects_non_integer_values(self):
        invalid_top_k_values = (True, 1.5, np.nan, np.inf, np.array([1]))

        for field_name in ("row_top_k", "column_top_k"):
            for value in invalid_top_k_values:
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaisesRegex(
                        ValueError,
                        f"{field_name} must be a positive integer or None",
                    ):
                        CandidatePruningConfig(**{field_name: value})


if __name__ == "__main__":
    unittest.main()
