import unittest

import numpy as np
from pyrecest.filters.track_manager import solve_global_nearest_neighbor


class TrackManagerCostValidationTest(unittest.TestCase):
    def test_solver_rejects_complex_cost_matrices(self):
        invalid_cost_matrices = (
            np.array([[0.0 + 2.0j, 10.0], [10.0, 0.0]]),
            np.array([[0.0 + 2.0j]], dtype=object),
        )

        for cost_matrix in invalid_cost_matrices:
            with self.subTest(dtype=cost_matrix.dtype):
                with self.assertRaisesRegex(ValueError, "cost_matrix.*numeric"):
                    solve_global_nearest_neighbor(
                        cost_matrix,
                        unassigned_track_cost=5.0,
                        unassigned_measurement_cost=5.0,
                    )

    def test_solver_rejects_complex_unassigned_cost_vectors(self):
        with self.assertRaisesRegex(ValueError, "unassigned_track_cost.*numeric"):
            solve_global_nearest_neighbor(
                np.array([[0.0, 10.0], [10.0, 0.0]]),
                unassigned_track_cost=np.array([5.0 + 3.0j, 5.0]),
                unassigned_measurement_cost=5.0,
            )

    def test_solver_rejects_masked_cost_inputs(self):
        masked_matrix = np.ma.array(
            [[0.0, 10.0], [10.0, 0.0]],
            mask=[[True, False], [False, False]],
        )
        with self.assertRaisesRegex(ValueError, "cost_matrix.*numeric"):
            solve_global_nearest_neighbor(
                masked_matrix,
                unassigned_track_cost=5.0,
                unassigned_measurement_cost=5.0,
            )

        masked_unassigned_cost = np.ma.array([5.0, 5.0], mask=[False, True])
        with self.assertRaisesRegex(ValueError, "unassigned_track_cost.*numeric"):
            solve_global_nearest_neighbor(
                np.array([[0.0, 10.0], [10.0, 0.0]]),
                unassigned_track_cost=masked_unassigned_cost,
                unassigned_measurement_cost=5.0,
            )

    def test_solver_accepts_masked_arrays_without_masked_entries(self):
        association = solve_global_nearest_neighbor(
            np.ma.array(
                [[0.0, 10.0], [10.0, 0.0]],
                mask=False,
            ),
            unassigned_track_cost=np.ma.array([5.0, 5.0], mask=False),
            unassigned_measurement_cost=5.0,
        )

        self.assertEqual(association.matches, [(0, 0), (1, 1)])
        self.assertEqual(association.unmatched_track_indices, [])
        self.assertEqual(association.unmatched_measurement_indices, [])


if __name__ == "__main__":
    unittest.main()
