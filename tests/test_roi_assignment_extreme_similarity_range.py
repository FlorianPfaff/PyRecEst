import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest import backend
from pyrecest.backend import array, float64
from pyrecest.utils.roi_assignment import assign_by_similarity_matrix


@unittest.skipIf(
    backend.__backend_name__ == "jax",
    reason="Not supported on the jax backend",
)
class TestExtremeSimilarityAssignment(unittest.TestCase):
    def test_assignment_handles_full_finite_float64_range(self):
        limit = np.finfo(np.float64).max
        similarities = array(
            [[limit, -limit], [-limit, limit]],
            dtype=float64,
        )

        result = assign_by_similarity_matrix(
            similarities,
            min_similarity=-limit,
            return_result=True,
        )

        npt.assert_array_equal(result.assignment, array([0, 1]))
        npt.assert_array_equal(
            result.matched_similarities,
            array([limit, limit], dtype=float64),
        )

    def test_assignment_preserves_small_scores_beside_extreme_values(self):
        limit = np.finfo(np.float64).max
        minimum = -1.0e300
        similarities = array(
            [
                [limit, minimum, minimum],
                [minimum, 0.0, 3.0e-100],
                [minimum, 2.0e-100, 0.0],
            ],
            dtype=float64,
        )

        assignment = assign_by_similarity_matrix(
            similarities,
            min_similarity=minimum,
        )

        npt.assert_array_equal(assignment, array([0, 2, 1]))

    def test_assignment_handles_dummy_cost_at_float64_limit(self):
        limit = np.finfo(np.float64).max

        result = assign_by_similarity_matrix(
            array([[limit]], dtype=float64),
            min_similarity=0.0,
            return_result=True,
        )

        npt.assert_array_equal(result.assignment, array([0]))
        npt.assert_array_equal(
            result.matched_similarities,
            array([limit], dtype=float64),
        )

    def test_assignment_returns_unmatched_above_extreme_score(self):
        limit = np.finfo(np.float64).max

        assignment = assign_by_similarity_matrix(
            array([[-limit]], dtype=float64),
            min_similarity=limit,
        )

        npt.assert_array_equal(assignment, array([-1]))


if __name__ == "__main__":
    unittest.main()
