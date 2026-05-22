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
        cfg = candidate_pruning_config_from_mapping({"row_top_k": 2})
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


if __name__ == "__main__":
    unittest.main()
