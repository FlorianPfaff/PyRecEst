import unittest

import numpy as np
import numpy.testing as npt

from pyrecest.utils.higher_order_assignment import (
    HigherOrderConsistencyConfig,
    apply_higher_order_consistency,
    higher_order_consistency_config_from_mapping,
    min_plus_triplet_support,
    triplet_consistency_penalty,
    triplet_support_costs,
)


class TestMinPlusTripletSupport(unittest.TestCase):
    def test_matches_dense_reference_with_forbidden_entries(self):
        left = np.array([[1.0, np.inf, 4.0], [3.0, 2.0, np.inf]])
        right = np.array([[5.0, 1.0], [2.0, 7.0], [0.5, np.inf]])

        support = min_plus_triplet_support(left, right)

        npt.assert_allclose(support, np.array([[4.5, 2.0], [4.0, 4.0]]))

    def test_respects_large_finite_forbidden_cost(self):
        support = min_plus_triplet_support(
            np.array([[0.2, 100.0]]),
            np.array([[0.3], [0.1]]),
            large_cost=10.0,
        )

        npt.assert_allclose(support, np.array([[0.5]]))

    def test_empty_shared_axis_returns_infinite_support(self):
        support = min_plus_triplet_support(
            np.empty((2, 0)),
            np.empty((0, 3)),
        )

        self.assertEqual(support.shape, (2, 3))
        self.assertTrue(np.all(np.isposinf(support)))

    def test_rejects_invalid_shapes_and_nonfinite_values(self):
        with self.assertRaisesRegex(ValueError, "left_costs must be two-dimensional"):
            min_plus_triplet_support(np.array([1.0]), np.ones((1, 1)))
        with self.assertRaisesRegex(ValueError, "columns must match"):
            min_plus_triplet_support(np.ones((2, 3)), np.ones((2, 4)))
        for invalid in (np.nan, -np.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "positive infinity"):
                    min_plus_triplet_support(
                        np.array([[invalid]]),
                        np.ones((1, 1)),
                    )


class TestHigherOrderAssignmentConsistency(unittest.TestCase):
    def test_bridge_context_supports_consistent_skip_edges(self):
        costs = {
            (0, 1): np.array([[0.1, np.inf], [np.inf, 0.1]]),
            (1, 2): np.array([[0.1, np.inf], [np.inf, 0.1]]),
            (0, 2): np.ones((2, 2)),
        }

        support = triplet_support_costs(costs, edge=(0, 2))

        npt.assert_allclose(
            support,
            np.array([[0.2, np.inf], [np.inf, 0.2]]),
        )

    def test_backward_and_forward_contexts_use_correct_orientation(self):
        backward_costs = {
            (0, 1): np.array([[0.1, np.inf], [np.inf, 0.1]]),
            (0, 2): np.array([[0.2, np.inf], [np.inf, 0.2]]),
            (1, 2): np.ones((2, 2)),
        }
        forward_costs = {
            (0, 1): np.ones((2, 2)),
            (0, 2): np.array([[0.1, np.inf], [np.inf, 0.1]]),
            (1, 2): np.array([[0.2, np.inf], [np.inf, 0.2]]),
        }

        backward = triplet_support_costs(backward_costs, edge=(1, 2))
        forward = triplet_support_costs(forward_costs, edge=(0, 1))

        expected = np.array([[0.3, np.inf], [np.inf, 0.3]])
        npt.assert_allclose(backward, expected)
        npt.assert_allclose(forward, expected)

    def test_best_available_context_is_used(self):
        costs = {
            (0, 1): np.array([[0.2]]),
            (0, 2): np.array([[0.2]]),
            (0, 3): np.array([[5.0]]),
            (1, 2): np.array([[10.0]]),
            (1, 3): np.array([[0.1]]),
            (2, 3): np.array([[0.1]]),
        }

        support = triplet_support_costs(costs, edge=(1, 2))

        npt.assert_allclose(support, np.array([[0.2]]))

    def test_nonconsecutive_session_labels_are_supported(self):
        costs = {
            (10, 20): np.array([[0.1]]),
            (20, 30): np.array([[0.2]]),
            (10, 30): np.array([[1.0]]),
        }

        support = triplet_support_costs(
            costs,
            edge=(10, 30),
            session_sizes={10: 1, 20: 1, 30: 1},
        )

        npt.assert_allclose(support, np.array([[0.3]]))

    def test_no_complete_context_returns_none(self):
        costs = {
            (0, 1): np.array([[1.0]]),
            (1, 2): np.array([[1.0]]),
        }

        self.assertIsNone(triplet_support_costs(costs, edge=(0, 1)))
        self.assertIsNone(triplet_consistency_penalty(costs, edge=(0, 1)))

    def test_bounded_penalty_and_adjustment_preserve_forbidden_edges(self):
        costs = {
            (0, 1): np.array([[0.1, np.inf], [np.inf, 0.1]]),
            (1, 2): np.array([[0.1, np.inf], [np.inf, 0.1]]),
            (0, 2): np.array([[1.0, np.inf], [1.0, 1.0e6]]),
        }
        config = HigherOrderConsistencyConfig(
            triplet_weight=2.0,
            support_cost_cap=0.5,
            max_penalty=3.0,
            large_cost=1.0e6,
        )

        penalty = triplet_consistency_penalty(
            costs,
            edge=(0, 2),
            config=config,
        )
        adjusted = apply_higher_order_consistency(costs, config=config)

        npt.assert_allclose(
            penalty,
            np.array([[0.0, 3.0], [3.0, 0.0]]),
        )
        npt.assert_allclose(
            adjusted[(0, 2)],
            np.array([[1.0, np.inf], [7.0, 1.0e6]]),
        )

    def test_penalty_is_monotone_in_support_cost_and_capped(self):
        direct = np.array([[1.0]])
        good_support = {
            (0, 1): np.array([[0.1]]),
            (1, 2): np.array([[0.1]]),
            (0, 2): direct,
        }
        weak_support = {
            (0, 1): np.array([[2.0]]),
            (1, 2): np.array([[2.0]]),
            (0, 2): direct,
        }
        config = HigherOrderConsistencyConfig(
            triplet_weight=1.0,
            support_cost_cap=0.5,
            max_penalty=1.5,
        )

        good_penalty = triplet_consistency_penalty(
            good_support, edge=(0, 2), config=config
        )
        weak_penalty = triplet_consistency_penalty(
            weak_support, edge=(0, 2), config=config
        )

        npt.assert_allclose(good_penalty, np.array([[0.0]]))
        npt.assert_allclose(weak_penalty, np.array([[1.5]]))
        self.assertLessEqual(good_penalty[0, 0], weak_penalty[0, 0])

    def test_disabled_config_returns_independent_copies(self):
        matrix = np.array([[1.0, 2.0]])
        costs = {(0, 1): matrix}

        adjusted = apply_higher_order_consistency(costs)

        npt.assert_allclose(adjusted[(0, 1)], matrix)
        self.assertIsNot(adjusted[(0, 1)], matrix)
        adjusted[(0, 1)][0, 0] = -1.0
        self.assertEqual(matrix[0, 0], 1.0)

    def test_shape_validation_checks_explicit_and_inferred_sizes(self):
        costs = {
            (0, 1): np.ones((2, 3)),
            (1, 2): np.ones((4, 1)),
        }

        with self.assertRaisesRegex(ValueError, "Session 1 has size 4"):
            apply_higher_order_consistency(costs)
        with self.assertRaisesRegex(ValueError, "session_sizes specifies 2"):
            apply_higher_order_consistency(
                {(0, 1): np.ones((2, 3))},
                session_sizes=(2, 2),
            )

    def test_missing_edge_and_invalid_edge_keys_are_rejected(self):
        with self.assertRaisesRegex(KeyError, "No pairwise cost matrix"):
            triplet_support_costs({(0, 1): np.ones((1, 1))}, edge=(0, 2))

        invalid_edges = ((1, 1), (-1, 1), (0,), "01")
        for invalid_edge in invalid_edges:
            with self.subTest(invalid_edge=invalid_edge):
                with self.assertRaises(ValueError):
                    apply_higher_order_consistency(
                        {invalid_edge: np.ones((1, 1))}
                    )

    def test_config_mapping_and_validation(self):
        config = higher_order_consistency_config_from_mapping(
            {
                "triplet_weight": np.array(0.25),
                "support_cost_cap": -1.0,
                "max_penalty": np.array(2.0),
            }
        )

        self.assertIsInstance(config, HigherOrderConsistencyConfig)
        self.assertEqual(config.triplet_weight, 0.25)
        self.assertEqual(config.support_cost_cap, -1.0)
        self.assertEqual(config.max_penalty, 2.0)

        invalid_configs = (
            {"triplet_weight": -1.0},
            {"triplet_weight": True},
            {"support_cost_cap": np.inf},
            {"max_penalty": -0.1},
            {"large_cost": 0.0},
            {"large_cost": "10"},
        )
        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    HigherOrderConsistencyConfig(**kwargs)

        with self.assertRaisesRegex(ValueError, "mapping, or None"):
            higher_order_consistency_config_from_mapping(1.0)


if __name__ == "__main__":
    unittest.main()
