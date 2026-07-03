import unittest

import numpy as np

from pyrecest.utils.multisession_assignment_problem import (
    MultiSessionAssignmentProblem,
    MultiSessionAssignmentRun,
    session_edge_pairs,
)


class TestMultiSessionAssignmentProblem(unittest.TestCase):
    def test_problem_solves_and_keeps_metadata(self):
        problem = MultiSessionAssignmentProblem(
            pairwise_costs=[np.array([[0.1, 5.0], [5.0, 0.2]])],
            start_cost=1.0,
            end_cost=1.0,
            metadata={"source": "synthetic"},
        )

        run = problem.solved()

        self.assertIs(run.problem, problem)
        self.assertEqual(run.problem.metadata["source"], "synthetic")
        self.assertEqual(run.n_matched_edges, 2)
        self.assertEqual(run.n_tracks, 2)
        self.assertEqual(
            [(edge[0], edge[1]) for edge in run.result.matched_edges],
            [((0, 0), (1, 0)), ((0, 1), (1, 1))],
        )
        self.assertAlmostEqual(run.result.matched_edges[0][2], 0.1)
        self.assertAlmostEqual(run.result.matched_edges[1][2], 0.2)

    def test_run_solve_classmethod_and_summary(self):
        problem = MultiSessionAssignmentProblem(
            pairwise_costs={(0, 1): np.array([[0.25]])},
            session_sizes=(1, 1),
            start_cost=1.0,
            end_cost=1.0,
            gap_penalty=0.5,
            cost_threshold=10.0,
        )

        run = MultiSessionAssignmentRun.solve(problem)
        summary = run.to_summary_dict()

        self.assertEqual(summary["n_tracks"], 1)
        self.assertEqual(summary["n_matched_edges"], 1)
        self.assertEqual(summary["start_cost"], 1.0)
        self.assertEqual(summary["end_cost"], 1.0)
        self.assertEqual(summary["gap_penalty"], 0.5)
        self.assertEqual(summary["cost_threshold"], 10.0)
        self.assertLess(summary["total_cost"], 2.0)

    def test_with_pairwise_costs_replaces_costs_and_updates_metadata(self):
        original = MultiSessionAssignmentProblem(
            pairwise_costs=[np.array([[3.0]])],
            metadata={"stage": "raw"},
        )

        updated = original.with_pairwise_costs(
            [np.array([[0.1]])],
            metadata_update={"stage": "adjusted", "adjustment": "triplet"},
        )

        self.assertEqual(original.metadata["stage"], "raw")
        self.assertEqual(updated.metadata["stage"], "adjusted")
        self.assertEqual(updated.metadata["adjustment"], "triplet")
        self.assertIsNot(original, updated)
        self.assertIs(updated.session_sizes, original.session_sizes)

    def test_session_edge_pairs_supports_skip_edges(self):
        self.assertEqual(session_edge_pairs(0), ())
        self.assertEqual(session_edge_pairs(1), ())
        self.assertEqual(session_edge_pairs(4), ((0, 1), (1, 2), (2, 3)))
        self.assertEqual(
            session_edge_pairs(4, max_gap=2),
            ((0, 1), (0, 2), (1, 2), (1, 3), (2, 3)),
        )

    def test_session_edge_pairs_validates_inputs(self):
        invalid_calls = (
            lambda: session_edge_pairs(-1),
            lambda: session_edge_pairs(3, max_gap=0),
            lambda: session_edge_pairs(3.5),
            lambda: session_edge_pairs(True),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()


if __name__ == "__main__":
    unittest.main()
