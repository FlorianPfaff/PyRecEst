import unittest

import numpy as np
import numpy.testing as npt

from pyrecest.utils.higher_order_assignment import (
    HigherOrderConsistencyConfig,
    apply_higher_order_consistency,
    higher_order_consistency_config_from_mapping,
    triplet_consistency_penalty,
)


class TestHigherOrderAssignmentConsistency(unittest.TestCase):
    def test_disabled_config_copies_costs_without_change(self):
        costs = {
            (0, 1): np.array([[1.0, 2.0], [3.0, 4.0]]),
            (1, 2): np.array([[0.5, 4.0], [4.0, 0.5]]),
        }

        adjusted = apply_higher_order_consistency(costs, session_sizes=(2, 2, 2))

        self.assertEqual(set(adjusted), set(costs))
        for edge, matrix in costs.items():
            npt.assert_allclose(adjusted[edge], matrix)
            self.assertIsNot(adjusted[edge], matrix)

    def test_bridge_support_penalizes_unsupported_skip_edges(self):
        costs = {
            (0, 1): np.array([[0.1, 10.0], [10.0, 0.1]]),
            (1, 2): np.array([[0.1, 10.0], [10.0, 0.1]]),
            (0, 2): np.ones((2, 2)),
        }
        config = HigherOrderConsistencyConfig(
            triplet_weight=2.0,
            support_top_k=1,
            support_cost_cap=0.5,
            max_penalty=3.0,
        )

        adjusted = apply_higher_order_consistency(
            costs,
            session_sizes=(2, 2, 2),
            config=config,
        )

        npt.assert_allclose(
            adjusted[(0, 2)],
            np.array([[1.0, 7.0], [7.0, 1.0]]),
        )
        npt.assert_allclose(adjusted[(0, 1)], costs[(0, 1)])
        npt.assert_allclose(adjusted[(1, 2)], costs[(1, 2)])

    def test_forward_context_penalizes_consecutive_edge_without_future_support(self):
        costs = {
            (0, 1): np.ones((2, 2)),
            (0, 2): np.array([[0.1, 10.0], [10.0, 0.1]]),
            (1, 2): np.array([[0.1, 10.0], [10.0, 0.1]]),
        }
        config = HigherOrderConsistencyConfig(
            triplet_weight=1.5,
            support_top_k=1,
            support_cost_cap=0.5,
            max_penalty=2.0,
        )

        penalty = triplet_consistency_penalty(
            costs,
            edge=(0, 1),
            session_sizes=(2, 2, 2),
            config=config,
        )
        adjusted = apply_higher_order_consistency(
            costs,
            session_sizes=(2, 2, 2),
            config=config,
        )

        npt.assert_allclose(penalty, np.array([[0.0, 2.0], [2.0, 0.0]]))
        npt.assert_allclose(
            adjusted[(0, 1)],
            np.array([[1.0, 4.0], [4.0, 1.0]]),
        )

    def test_backward_context_penalizes_edge_without_past_support(self):
        costs = {
            (0, 1): np.array([[0.1, 10.0], [10.0, 0.1]]),
            (0, 2): np.array([[0.1, 10.0], [10.0, 0.1]]),
            (1, 2): np.ones((2, 2)),
        }
        config = HigherOrderConsistencyConfig(
            triplet_weight=1.0,
            support_top_k=1,
            support_cost_cap=0.5,
            max_penalty=2.5,
        )

        adjusted = apply_higher_order_consistency(
            costs,
            session_sizes=(2, 2, 2),
            config=config,
        )

        npt.assert_allclose(
            adjusted[(1, 2)],
            np.array([[1.0, 3.5], [3.5, 1.0]]),
        )

    def test_large_cost_entries_are_not_adjusted(self):
        costs = {
            (0, 1): np.array([[0.1]]),
            (1, 2): np.array([[0.1]]),
            (0, 2): np.array([[1.0e6]]),
        }
        config = HigherOrderConsistencyConfig(
            triplet_weight=5.0,
            support_cost_cap=0.0,
            max_penalty=5.0,
            large_cost=1.0e6,
        )

        adjusted = apply_higher_order_consistency(
            costs,
            session_sizes=(1, 1, 1),
            config=config,
        )

        self.assertEqual(adjusted[(0, 2)][0, 0], 1.0e6)

    def test_shape_validation_rejects_mismatched_pairwise_costs(self):
        costs = {(0, 1): np.ones((2, 3))}

        with self.assertRaisesRegex(ValueError, "expected \(2, 2\)"):
            apply_higher_order_consistency(costs, session_sizes=(2, 2))

    def test_config_mapping_and_validation(self):
        cfg = higher_order_consistency_config_from_mapping(
            {"triplet_weight": np.array(0.25), "support_top_k": np.array(2.0)}
        )
        self.assertIsInstance(cfg, HigherOrderConsistencyConfig)
        self.assertEqual(cfg.support_top_k, 2)
        self.assertEqual(cfg.triplet_weight, 0.25)

        invalid_configs = (
            {"triplet_weight": -1.0},
            {"support_top_k": 0},
            {"support_cost_cap": np.inf},
            {"max_penalty": True},
            {"large_cost": 0.0},
        )
        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    HigherOrderConsistencyConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
