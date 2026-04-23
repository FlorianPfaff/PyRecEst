"""Tests for observation-specific multi-session assignment costs."""

import unittest

from pyrecest.backend import (  # pylint: disable=no-name-in-module
    __backend_name__,
    array,
)
from pyrecest.utils import (
    solve_multisession_assignment,
    solve_multisession_assignment_with_observation_costs,
)


@unittest.skipIf(
    __backend_name__ == "jax",
    reason="Not supported on this backend",
)
class TestMultiSessionAssignmentObservationCosts(unittest.TestCase):
    @staticmethod
    def _canonical_tracks(tracks):
        return sorted(tuple(sorted(track.items())) for track in tracks)

    def test_matches_base_solver_when_costs_are_uniform(self):
        pairwise_costs = [
            array([[0.1, 8.0], [8.0, 0.2]], dtype=float),
            array([[0.1, 8.0], [8.0, 0.2]], dtype=float),
        ]
        expected = solve_multisession_assignment(
            pairwise_costs,
            start_cost=5.0,
            end_cost=5.0,
            cost_threshold=10.0,
        )
        actual = solve_multisession_assignment_with_observation_costs(
            pairwise_costs,
            start_cost=5.0,
            end_cost=5.0,
            cost_threshold=10.0,
        )

        self.assertEqual(
            self._canonical_tracks(actual.tracks),
            self._canonical_tracks(expected.tracks),
        )
        self.assertEqual(actual.matched_edges, expected.matched_edges)
        self.assertAlmostEqual(actual.total_cost, expected.total_cost)

    def test_detection_specific_start_costs_bias_target_choice(self):
        result = solve_multisession_assignment_with_observation_costs(
            [array([[1.5, 1.5]], dtype=float)],
            start_cost=0.0,
            end_cost=2.0,
            start_costs={1: array([0.0, 3.0], dtype=float)},
        )

        expected_tracks = [
            ((0, 0), (1, 1)),
            ((1, 0),),
        ]
        self.assertEqual(self._canonical_tracks(result.tracks), expected_tracks)
        self.assertEqual(result.matched_edges, [((0, 0), (1, 1), 1.5)])
        self.assertAlmostEqual(result.total_cost, 5.5)

    def test_detection_specific_end_costs_bias_source_choice(self):
        result = solve_multisession_assignment_with_observation_costs(
            [array([[1.5], [1.5]], dtype=float)],
            start_cost=2.0,
            end_cost=0.0,
            end_costs={0: array([0.0, 3.0], dtype=float)},
        )

        expected_tracks = [
            ((0, 0),),
            ((0, 1), (1, 0)),
        ]
        self.assertEqual(self._canonical_tracks(result.tracks), expected_tracks)
        self.assertEqual(result.matched_edges, [((0, 1), (1, 0), 1.5)])
        self.assertAlmostEqual(result.total_cost, 5.5)

    def test_sequence_scalar_cost_entries_are_broadcast_per_session(self):
        result = solve_multisession_assignment_with_observation_costs(
            [array([[2.5, 2.5]], dtype=float)],
            start_cost=0.0,
            end_cost=1.0,
            start_costs=[0.0, 3.0],
        )

        self.assertEqual(len(result.matched_edges), 1)
        self.assertAlmostEqual(result.total_cost, 7.5)

    def test_cost_threshold_is_applied_in_original_cost_domain(self):
        result = solve_multisession_assignment_with_observation_costs(
            {(0, 2): array([[0.3]], dtype=float)},
            session_sizes=[1, 0, 1],
            start_cost=4.0,
            end_cost=4.0,
            start_costs={2: array([10.0])},
            gap_penalty=0.5,
            cost_threshold=0.75,
        )

        self.assertEqual(
            self._canonical_tracks(result.tracks),
            [((0, 0),), ((2, 0),)],
        )
        self.assertEqual(result.matched_edges, [])

    def test_rejects_unknown_sessions(self):
        with self.assertRaises(ValueError):
            solve_multisession_assignment_with_observation_costs(
                [array([[1.0]], dtype=float)],
                start_costs={2: array([1.0], dtype=float)},
            )

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            solve_multisession_assignment_with_observation_costs(
                [array([[1.0, 1.0]], dtype=float)],
                start_costs={1: array([1.0], dtype=float)},
            )


if __name__ == "__main__":
    unittest.main()
