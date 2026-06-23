"""Regression tests for observation-cost assignment gap handling."""

import unittest

from pyrecest.backend import (  # pylint: disable=no-name-in-module
    __backend_name__,
    array,
)
from pyrecest.utils import solve_multisession_assignment_with_observation_costs


@unittest.skipIf(
    __backend_name__ == "jax",
    reason="Not supported on this backend",
)
class TestMultiSessionAssignmentObservationCostGaps(unittest.TestCase):
    @staticmethod
    def _canonical_tracks(tracks):
        return sorted(tuple(sorted(track.items())) for track in tracks)

    def test_gap_penalty_uses_numeric_session_indices_when_sizes_are_inferred(self):
        result = solve_multisession_assignment_with_observation_costs(
            {(0, 2): array([[0.3]], dtype=float)},
            start_cost=4.0,
            end_cost=4.0,
            gap_penalty=0.5,
        )

        self.assertEqual(
            self._canonical_tracks(result.tracks),
            [((0, 0), (2, 0))],
        )
        self.assertEqual(result.matched_edges, [((0, 0), (2, 0), 0.8)])
        self.assertAlmostEqual(result.total_cost, 8.8)

    def test_cost_threshold_uses_numeric_session_gap_when_sizes_are_inferred(self):
        result = solve_multisession_assignment_with_observation_costs(
            {(0, 2): array([[0.3]], dtype=float)},
            start_cost=4.0,
            end_cost=4.0,
            gap_penalty=0.5,
            cost_threshold=0.75,
        )

        self.assertEqual(
            self._canonical_tracks(result.tracks),
            [((0, 0),), ((2, 0),)],
        )
        self.assertEqual(result.matched_edges, [])
        self.assertAlmostEqual(result.total_cost, 16.0)


if __name__ == "__main__":
    unittest.main()
