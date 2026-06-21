"""Regression tests for score-native multi-session assignment gap handling."""

import unittest

from pyrecest.backend import (  # pylint: disable=no-name-in-module
    __backend_name__,
    array,
)
from pyrecest.utils import solve_multisession_assignment_from_similarity


class TestMultiSessionAssignmentSimilarityGap(unittest.TestCase):
    @staticmethod
    def _canonical_tracks(tracks):
        return sorted(tuple(sorted(track.items())) for track in tracks)

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_max_gap_uses_numeric_session_indices_when_sizes_are_inferred(self):
        result = solve_multisession_assignment_from_similarity(
            {(0, 2): array([[0.95]], dtype=float)},
            max_gap=0,
            start_cost=1.0,
            end_cost=1.0,
        )

        self.assertEqual(
            self._canonical_tracks(result.tracks),
            [((0, 0),), ((2, 0),)],
        )
        self.assertEqual(result.matched_edges, [])
        self.assertAlmostEqual(result.total_cost, 4.0)


if __name__ == "__main__":
    unittest.main()
