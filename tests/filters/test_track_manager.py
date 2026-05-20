import unittest

import numpy as np

from pyrecest.filters.track_manager import (
    AssociationResult,
    TrackManager,
    solve_global_nearest_neighbor,
)


class TrackManagerAssociationTest(unittest.TestCase):
    def test_normalize_association_result_requires_track_coverage(self):
        with self.assertRaisesRegex(ValueError, "track"):
            TrackManager._normalize_association_result(
                AssociationResult(
                    matches=[],
                    unmatched_track_indices=[],
                    unmatched_measurement_indices=[0],
                ),
                num_tracks=1,
                num_measurements=1,
            )

    def test_normalize_association_result_requires_measurement_coverage(self):
        with self.assertRaisesRegex(ValueError, "measurement"):
            TrackManager._normalize_association_result(
                AssociationResult(
                    matches=[],
                    unmatched_track_indices=[0],
                    unmatched_measurement_indices=[],
                ),
                num_tracks=1,
                num_measurements=1,
            )

    def test_solver_does_not_return_nonfinite_pair_as_match(self):
        association = solve_global_nearest_neighbor(
            np.array([[np.inf]]),
            unassigned_track_cost=np.inf,
            unassigned_measurement_cost=np.inf,
        )

        self.assertEqual(association.matches, [])
        self.assertEqual(association.unmatched_track_indices, [0])
        self.assertEqual(association.unmatched_measurement_indices, [0])
