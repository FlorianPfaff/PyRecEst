"""Tests for global multi-session association."""

import unittest
from unittest.mock import patch

import numpy as np
from pyrecest.backend import (  # pylint: disable=no-name-in-module
    __backend_name__,
    array,
    array_equal,
)
from pyrecest.utils import multisession_assignment as multisession_assignment_module
from pyrecest.utils import (
    multisession_assignment_score as multisession_assignment_score_module,
)
from pyrecest.utils import (
    solve_multisession_assignment,
    solve_multisession_assignment_from_similarity,
    tracks_to_session_labels,
)


class TestMultiSessionAssignment(unittest.TestCase):
    @staticmethod
    def _canonical_tracks(tracks):
        return sorted(tuple(sorted(track.items())) for track in tracks)

    def test_solve_multisession_assignment_rejects_jax_backend(self):
        with patch.object(multisession_assignment_module, "__backend_name__", "jax"):
            with self.assertRaisesRegex(NotImplementedError, "JAX backend"):
                solve_multisession_assignment({}, session_sizes=[1])

    def test_tracks_to_session_labels_rejects_jax_backend(self):
        with patch.object(multisession_assignment_module, "__backend_name__", "jax"):
            with self.assertRaisesRegex(NotImplementedError, "JAX backend"):
                tracks_to_session_labels([{0: 0}])

    def test_similarity_wrapper_rejects_jax_backend(self):
        with patch.object(
            multisession_assignment_score_module,
            "__backend_name__",
            "jax",
        ):
            with self.assertRaisesRegex(NotImplementedError, "JAX backend"):
                solve_multisession_assignment_from_similarity([array([[1.0]])])

    def test_tracks_to_index_matrix_rejects_jax_backend(self):
        with patch.object(
            multisession_assignment_score_module,
            "__backend_name__",
            "jax",
        ):
            with self.assertRaisesRegex(NotImplementedError, "JAX backend"):
                multisession_assignment_score_module.tracks_to_index_matrix([{0: 0}])

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_singletons_without_edges(self):
        result = solve_multisession_assignment(
            {},
            session_sizes=[2, 1, 2],
            start_cost=1.0,
            end_cost=1.0,
        )

        expected_tracks = [
            ((0, 0),),
            ((0, 1),),
            ((1, 0),),
            ((2, 0),),
            ((2, 1),),
        ]
        self.assertEqual(self._canonical_tracks(result.tracks), expected_tracks)
        self.assertAlmostEqual(result.total_cost, 10.0)
        self.assertEqual(result.matched_edges, [])

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_consecutive_costs_form_two_long_tracks(self):
        result = solve_multisession_assignment(
            [
                array([[0.1, 8.0], [8.0, 0.2]], dtype=float),
                array([[0.1, 8.0], [8.0, 0.2]], dtype=float),
            ],
            start_cost=5.0,
            end_cost=5.0,
            cost_threshold=10.0,
        )

        expected_tracks = [
            ((0, 0), (1, 0), (2, 0)),
            ((0, 1), (1, 1), (2, 1)),
        ]
        self.assertEqual(self._canonical_tracks(result.tracks), expected_tracks)
        self.assertAlmostEqual(result.total_cost, 20.6)

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_cross_gap_linking_is_supported(self):
        result = solve_multisession_assignment(
            {(0, 2): array([[0.3]], dtype=float)},
            session_sizes=[1, 0, 1],
            start_cost=4.0,
            end_cost=4.0,
            gap_penalty=0.5,
        )

        expected_tracks = [((0, 0), (2, 0))]
        self.assertEqual(self._canonical_tracks(result.tracks), expected_tracks)
        self.assertAlmostEqual(result.total_cost, 8.8)
        self.assertEqual(result.matched_edges, [((0, 0), (2, 0), 0.8)])

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_gap_penalty_uses_numeric_session_indices_when_sizes_are_inferred(self):
        result = solve_multisession_assignment(
            {(0, 2): array([[0.3]], dtype=float)},
            start_cost=4.0,
            end_cost=4.0,
            gap_penalty=0.5,
        )

        expected_tracks = [((0, 0), (2, 0))]
        self.assertEqual(self._canonical_tracks(result.tracks), expected_tracks)
        self.assertAlmostEqual(result.total_cost, 8.8)
        self.assertEqual(result.matched_edges, [((0, 0), (2, 0), 0.8)])

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_similarity_wrapper_accepts_scalar_integer_like_max_gap(self):
        result = solve_multisession_assignment_from_similarity(
            {(0, 2): array([[0.9]], dtype=float)},
            session_sizes=[1, 0, 1],
            max_gap=np.array(1.0),
            start_cost=1.0,
            end_cost=1.0,
        )

        self.assertEqual(
            self._canonical_tracks(result.tracks),
            [((0, 0), (2, 0))],
        )

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_similarity_wrapper_rejects_invalid_max_gap(self):
        pairwise_scores = {(0, 2): array([[0.9]], dtype=float)}

        invalid_max_gaps = (True, 1.5, np.nan, np.inf, -1, np.array([1]))
        for max_gap in invalid_max_gaps:
            with self.subTest(max_gap=max_gap):
                with self.assertRaisesRegex(
                    ValueError,
                    "max_gap must be a non-negative integer",
                ):
                    solve_multisession_assignment_from_similarity(
                        pairwise_scores,
                        session_sizes=[1, 0, 1],
                        max_gap=max_gap,
                    )

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_global_solution_beats_pairwise_greedy_choice(self):
        result = solve_multisession_assignment(
            [
                array([[0.0, 1.0], [1.0, 100.0]], dtype=float),
                array([[100.0, 0.0], [0.0, 100.0]], dtype=float),
            ],
            start_cost=10.0,
            end_cost=10.0,
            cost_threshold=150.0,
        )

        expected_tracks = [
            ((0, 0), (1, 1), (2, 0)),
            ((0, 1), (1, 0), (2, 1)),
        ]
        self.assertEqual(self._canonical_tracks(result.tracks), expected_tracks)
        self.assertAlmostEqual(result.total_cost, 42.0)

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_tracks_to_session_labels(self):
        result = solve_multisession_assignment(
            {(0, 2): array([[0.3]], dtype=float)},
            session_sizes=[1, 0, 1],
            start_cost=4.0,
            end_cost=4.0,
            gap_penalty=0.5,
        )

        labels = result.to_session_labels(session_sizes=[1, 0, 1])
        self.assertTrue(array_equal(labels[0], array([0])))
        self.assertEqual(labels[1].shape[0], 0)
        self.assertTrue(array_equal(labels[2], array([0])))

        labels_from_function = tracks_to_session_labels(
            result.tracks, session_sizes=[1, 0, 1]
        )
        self.assertTrue(array_equal(labels_from_function[0], array([0])))
        self.assertEqual(labels_from_function[1].shape[0], 0)
        self.assertTrue(array_equal(labels_from_function[2], array([0])))

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_tracks_to_session_labels_rejects_duplicate_sessions_in_one_track(self):
        with self.assertRaisesRegex(
            ValueError,
            "Each track can only contain one detection per session",
        ):
            tracks_to_session_labels([[(0, 0), (0, 1)]], session_sizes=[2])

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_rejects_inconsistent_session_sizes(self):
        pairwise_costs = {
            (0, 1): array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            (1, 2): array([[0.0], [0.0], [0.0], [0.0]]),
        }

        with self.assertRaises(ValueError):
            solve_multisession_assignment(pairwise_costs)

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_rejects_negative_pairwise_session_indices(self):
        with self.assertRaisesRegex(
            ValueError,
            "Session indices must be non-negative",
        ):
            solve_multisession_assignment(
                {(-1, 0): array([[0.1]], dtype=float)},
                start_cost=1.0,
                end_cost=1.0,
            )

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_rejects_negative_explicit_session_indices(self):
        with self.assertRaisesRegex(
            ValueError,
            "Session indices must be non-negative",
        ):
            solve_multisession_assignment({}, session_sizes={-1: 1})

        with self.assertRaisesRegex(
            ValueError,
            "Session indices must be non-negative",
        ):
            tracks_to_session_labels([{-1: 0}])

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_drops_non_beneficial_links_even_without_threshold(self):
        result = solve_multisession_assignment(
            [array([[5.0]], dtype=float)],
            start_cost=1.0,
            end_cost=1.0,
        )

        self.assertEqual(self._canonical_tracks(result.tracks), [((0, 0),), ((1, 0),)])
        self.assertAlmostEqual(result.total_cost, 4.0)

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_sparse_backend_matches_previous_linprog_backend(self):
        pairwise_costs = [
            array(
                [
                    [0.4, 1.2, 7.5],
                    [2.0, 0.3, 3.5],
                    [6.0, 4.0, 0.2],
                ],
                dtype=float,
            ),
            array(
                [
                    [0.1, 3.0, 5.0],
                    [2.5, 0.2, 1.0],
                    [4.0, 1.5, 0.3],
                ],
                dtype=float,
            ),
        ]

        sparse_result = solve_multisession_assignment(
            pairwise_costs,
            start_cost=5.0,
            end_cost=5.0,
            cost_threshold=10.0,
        )
        with patch.object(
            multisession_assignment_module,
            "min_weight_full_bipartite_matching",
            side_effect=ValueError("forced sparse-backend failure"),
        ):
            linprog_result = solve_multisession_assignment(
                pairwise_costs,
                start_cost=5.0,
                end_cost=5.0,
                cost_threshold=10.0,
            )

        self.assertEqual(
            self._canonical_tracks(sparse_result.tracks),
            self._canonical_tracks(linprog_result.tracks),
        )
        self.assertEqual(sparse_result.matched_edges, linprog_result.matched_edges)
        self.assertAlmostEqual(sparse_result.total_cost, linprog_result.total_cost)

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_sparse_backend_falls_back_to_linprog_when_requested(self):
        pairwise_costs = [
            array([[0.1, 8.0], [8.0, 0.2]], dtype=float),
            array([[0.1, 8.0], [8.0, 0.2]], dtype=float),
        ]

        with patch.object(
            multisession_assignment_module,
            "min_weight_full_bipartite_matching",
            side_effect=ValueError("forced sparse-backend failure"),
        ):
            result = solve_multisession_assignment(
                pairwise_costs,
                start_cost=5.0,
                end_cost=5.0,
                cost_threshold=10.0,
            )

        expected_tracks = [
            ((0, 0), (1, 0), (2, 0)),
            ((0, 1), (1, 1), (2, 1)),
        ]
        self.assertEqual(self._canonical_tracks(result.tracks), expected_tracks)
        self.assertAlmostEqual(result.total_cost, 20.6)


if __name__ == "__main__":
    unittest.main()
