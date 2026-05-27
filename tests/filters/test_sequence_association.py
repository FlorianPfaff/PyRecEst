import unittest

import numpy as np
from pyrecest.filters import (
    SequenceAssociationNode,
    solve_top_k_viterbi_sequence_associations,
    solve_viterbi_sequence_association,
)


def _node(frame_index, candidate_index, track_id, unary_cost):
    return SequenceAssociationNode(
        frame_index=frame_index,
        candidate_index=candidate_index,
        unary_cost=unary_cost,
        payload=track_id,
    )


class SequenceAssociationTest(unittest.TestCase):
    def test_viterbi_prefers_sequence_cost_over_framewise_minimum(self):
        frames = [
            [_node(0, 0, "A", 0.0), _node(0, 1, "B", 0.9)],
            [_node(1, 0, "A", 1.5), _node(1, 1, "B", 0.0)],
            [_node(2, 0, "A", 0.0), _node(2, 1, "B", 0.9)],
        ]

        def transition_cost(previous, current, _context):
            return 0.0 if previous.payload == current.payload else 1.0

        result = solve_viterbi_sequence_association(frames, transition_cost)

        self.assertEqual(result.candidate_indices, (0, 0, 0))
        self.assertEqual(result.payloads, ("A", "A", "A"))
        self.assertAlmostEqual(result.total_cost, 1.5)

    def test_missed_detection_streak_is_available_to_transition_cost(self):
        frames = [
            [SequenceAssociationNode.missed_detection(0)],
            [SequenceAssociationNode.missed_detection(1)],
            [_node(2, 0, "A", 0.0)],
        ]
        observed_streaks = []

        def transition_cost(_previous, current, context):
            observed_streaks.append(context.previous_miss_streak)
            if current.is_missed_detection:
                return 1.0
            return 0.5 * context.previous_miss_streak

        result = solve_viterbi_sequence_association(frames, transition_cost)

        self.assertEqual(observed_streaks, [1, 2])
        self.assertEqual(result.missed_detection_frame_indices, (0, 1))
        self.assertEqual(result.transition_costs, (1.0, 1.0))
        self.assertAlmostEqual(result.total_cost, 2.0)

    def test_top_k_terminal_paths_are_sorted(self):
        frames = [
            [_node(0, 0, "A", 0.0), _node(0, 1, "B", 0.4)],
            [_node(1, 0, "A", 0.0), _node(1, 1, "B", 0.4)],
        ]

        def transition_cost(previous, current, _context):
            return 0.0 if previous.payload == current.payload else 5.0

        paths = solve_top_k_viterbi_sequence_associations(
            frames,
            transition_cost,
            top_k_terminal_paths=2,
        )

        self.assertEqual([path.payloads for path in paths], [("A", "A"), ("B", "B")])
        self.assertEqual([path.candidate_indices for path in paths], [(0, 0), (1, 1)])
        self.assertLessEqual(paths[0].total_cost, paths[1].total_cost)

    def test_validation_rejects_empty_frames_and_invalid_top_k(self):
        with self.assertRaises(ValueError):
            solve_viterbi_sequence_association([], lambda _a, _b, _c: 0.0)

        with self.assertRaises(ValueError):
            solve_top_k_viterbi_sequence_associations(
                [[_node(0, 0, "A", 0.0)]],
                lambda _a, _b, _c: 0.0,
                top_k_terminal_paths=0,
            )

        with self.assertRaises(ValueError):
            SequenceAssociationNode(0, None)

    def test_validation_rejects_nonfinite_unary_costs(self):
        for invalid_cost in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid_cost=invalid_cost):
                with self.assertRaises(ValueError):
                    _node(0, 0, "A", invalid_cost)

    def test_validation_rejects_nonfinite_transition_costs(self):
        frames = [[_node(0, 0, "A", 0.0)], [_node(1, 0, "A", 0.0)]]

        for invalid_cost in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid_cost=invalid_cost):
                with self.assertRaises(ValueError):
                    solve_viterbi_sequence_association(
                        frames, lambda _a, _b, _c: invalid_cost
                    )


if __name__ == "__main__":
    unittest.main()
