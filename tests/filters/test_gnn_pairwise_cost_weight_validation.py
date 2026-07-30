import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.filters.global_nearest_neighbor import GlobalNearestNeighbor


class GlobalNearestNeighborPairwiseCostWeightValidationTest(unittest.TestCase):
    def test_rejects_invalid_pairwise_cost_weights(self):
        invalid_weights = (
            True,
            "1.0",
            1.0 + 0.0j,
            np.nan,
            np.inf,
            -np.inf,
            -1.0,
            np.array([1.0]),
            np.datetime64("2026-07-28"),
            np.ma.masked,
        )

        for invalid_weight in invalid_weights:
            with self.subTest(pairwise_cost_weight=invalid_weight):
                tracker = GlobalNearestNeighbor(
                    association_param={"pairwise_cost_weight": invalid_weight}
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "pairwise_cost_weight must be a finite non-negative real scalar",
                ):
                    tracker._apply_pairwise_cost_matrix(
                        np.zeros((1, 1)), np.ones((1, 1))
                    )

    def test_accepts_finite_nonnegative_scalar_weights(self):
        for valid_weight in (0, 0.5, np.float64(2.0)):
            with self.subTest(pairwise_cost_weight=valid_weight):
                validated_weight = GlobalNearestNeighbor._validate_pairwise_cost_weight(
                    valid_weight
                )
                self.assertEqual(validated_weight, float(valid_weight))

    def test_zero_weight_ignores_positive_infinite_pairwise_gate(self):
        tracker = GlobalNearestNeighbor(association_param={"pairwise_cost_weight": 0.0})
        geometric_costs = np.array([[1.25]])

        combined_costs = tracker._apply_pairwise_cost_matrix(
            geometric_costs, np.array([[np.inf]])
        )

        npt.assert_array_equal(combined_costs, geometric_costs)

    def test_positive_weight_scales_pairwise_costs(self):
        tracker = GlobalNearestNeighbor(association_param={"pairwise_cost_weight": 2.0})

        combined_costs = tracker._apply_pairwise_cost_matrix(
            np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]])
        )

        npt.assert_array_equal(combined_costs, np.array([[7.0, 10.0]]))


if __name__ == "__main__":
    unittest.main()
