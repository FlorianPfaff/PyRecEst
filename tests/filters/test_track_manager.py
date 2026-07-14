import unittest

import numpy as np
from pyrecest.filters.track_manager import (
    AssociationResult,
    TrackManager,
    solve_global_nearest_neighbor,
)


class TrackManagerAssociationTest(unittest.TestCase):
    def test_lifecycle_thresholds_accept_integer_like_values(self):
        manager = TrackManager(n_init=np.array(2.0), max_misses=np.array(1.0))

        self.assertEqual(manager.n_init, 2)
        self.assertEqual(manager.max_misses, 1)

    def test_lifecycle_thresholds_reject_invalid_integer_values(self):
        invalid_n_init_values = (
            0,
            1.5,
            np.nan,
            np.inf,
            True,
            np.array([1]),
            "2",
            b"2",
            np.str_("2"),
        )
        for n_init in invalid_n_init_values:
            with self.subTest(n_init=n_init):
                with self.assertRaisesRegex(ValueError, "n_init must be"):
                    TrackManager(n_init=n_init)

        invalid_max_misses_values = (
            -1,
            1.5,
            np.nan,
            np.inf,
            False,
            np.array([1]),
            "1",
            b"1",
            np.str_("1"),
        )
        for max_misses in invalid_max_misses_values:
            with self.subTest(max_misses=max_misses):
                with self.assertRaisesRegex(ValueError, "max_misses must be"):
                    TrackManager(max_misses=max_misses)

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

    def test_normalize_association_result_rejects_noninteger_match_indices(self):
        invalid_associations = (
            (
                AssociationResult(matches=[(0.5, 0)]),
                "Track index must be a non-negative integer",
            ),
            (
                AssociationResult(matches=[(0, True)]),
                "Measurement index must be a non-negative integer",
            ),
            (
                AssociationResult(matches=[("0", 0)]),
                "Track index must be a non-negative integer",
            ),
            (
                AssociationResult(matches=[(0, np.str_("0"))]),
                "Measurement index must be a non-negative integer",
            ),
            (
                AssociationResult(
                    matches=[],
                    unmatched_track_indices=[0.5],
                    unmatched_measurement_indices=[0],
                ),
                "Unmatched track index must be a non-negative integer",
            ),
            (
                AssociationResult(
                    matches=[],
                    unmatched_track_indices=[0],
                    unmatched_measurement_indices=[np.array([0])],
                ),
                "Unmatched measurement index must be a non-negative integer",
            ),
            (
                AssociationResult(
                    matches=[],
                    unmatched_track_indices=["0"],
                    unmatched_measurement_indices=[0],
                ),
                "Unmatched track index must be a non-negative integer",
            ),
            (
                AssociationResult(
                    matches=[],
                    unmatched_track_indices=[0],
                    unmatched_measurement_indices=[b"0"],
                ),
                "Unmatched measurement index must be a non-negative integer",
            ),
        )

        for association, message in invalid_associations:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    TrackManager._normalize_association_result(
                        association,
                        num_tracks=1,
                        num_measurements=1,
                    )

    def test_solver_accepts_zero_dimensional_scalar_unassigned_costs(self):
        association = solve_global_nearest_neighbor(
            np.array([[0.0, 10.0], [10.0, 0.0]]),
            unassigned_track_cost=np.array(5.0),
            unassigned_measurement_cost=np.array(5.0),
        )

        self.assertEqual(association.matches, [(0, 0), (1, 1)])
        self.assertEqual(association.unmatched_track_indices, [])
        self.assertEqual(association.unmatched_measurement_indices, [])

    def test_solver_does_not_return_nonfinite_pair_as_match(self):
        association = solve_global_nearest_neighbor(
            np.array([[np.inf]]),
            unassigned_track_cost=np.inf,
            unassigned_measurement_cost=np.inf,
        )

        self.assertEqual(association.matches, [])
        self.assertEqual(association.unmatched_track_indices, [0])
        self.assertEqual(association.unmatched_measurement_indices, [0])
