import unittest

import numpy as np
from pyrecest.backend import __backend_name__, array
from pyrecest.utils import multisession_assignment_score as score_module
from pyrecest.utils import solve_multisession_assignment_from_similarity


class TestMultiSessionAssignmentScoreValidation(unittest.TestCase):
    @staticmethod
    def _text_scalars(text: str):
        return (text, text.encode(), np.str_(text), np.array(text))

    @staticmethod
    def _temporal_scalars():
        return (
            np.datetime64("1970-01-01T00:00:00.000000001"),
            np.timedelta64(1, "ns"),
            np.array(np.datetime64("1970-01-01T00:00:00.000000001")),
            np.array(np.timedelta64(1, "ns")),
            np.array(np.datetime64("1970-01-01T00:00:00.000000001"), dtype=object),
            np.array(np.timedelta64(1, "ns"), dtype=object),
        )

    @staticmethod
    def _negative_temporal_scalars():
        return (
            np.timedelta64(-1, "ns"),
            np.array(np.timedelta64(-1, "ns")),
            np.array(np.timedelta64(-1, "ns"), dtype=object),
        )

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_similarity_wrapper_rejects_text_scalar_min_score(self):
        pairwise_scores = {(0, 1): array([[0.9]], dtype=float)}

        for min_score in self._text_scalars("0.5"):
            with self.subTest(min_score=min_score):
                with self.assertRaisesRegex(
                    ValueError,
                    "min_score must be a finite scalar",
                ):
                    solve_multisession_assignment_from_similarity(
                        pairwise_scores,
                        session_sizes=[1, 1],
                        min_score=min_score,
                    )

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_similarity_wrapper_rejects_temporal_scalar_min_score(self):
        pairwise_scores = {(0, 1): array([[0.9]], dtype=float)}

        for min_score in self._temporal_scalars():
            with self.subTest(min_score=min_score):
                with self.assertRaisesRegex(
                    ValueError,
                    "min_score must be a finite scalar",
                ):
                    solve_multisession_assignment_from_similarity(
                        pairwise_scores,
                        session_sizes=[1, 1],
                        min_score=min_score,
                    )

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_similarity_wrapper_rejects_text_scalar_max_gap(self):
        pairwise_scores = {(0, 2): array([[0.9]], dtype=float)}

        for max_gap in self._text_scalars("1"):
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
    def test_similarity_wrapper_rejects_temporal_scalar_max_gap(self):
        pairwise_scores = {(0, 2): array([[0.9]], dtype=float)}

        for max_gap in self._temporal_scalars():
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
    def test_similarity_wrapper_rejects_non_real_score_matrices(self):
        invalid_pairwise_scores = (
            {(0, 1): [[True]]},
            {(0, 1): np.array([[True]], dtype=bool)},
            {(0, 1): [["0.9"]]},
            {(0, 1): np.array([["0.9"]])},
            {(0, 1): np.array([[0.9 + 0.0j]])},
            {(0, 1): np.array([[None]], dtype=object)},
            {(0, 1): np.array([[object()]], dtype=object)},
            {(0, 1): np.array([[{"score": 0.9}]], dtype=object)},
        )

        for pairwise_scores in invalid_pairwise_scores:
            with self.subTest(pairwise_scores=pairwise_scores):
                with self.assertRaisesRegex(
                    ValueError,
                    "pairwise_scores must contain real numeric score matrices",
                ):
                    solve_multisession_assignment_from_similarity(
                        pairwise_scores,
                        session_sizes=[1, 1],
                    )

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_similarity_wrapper_rejects_non_numeric_score_to_cost_object_entries(self):
        pairwise_scores = {(0, 1): array([[0.9]], dtype=float)}

        def score_to_cost(_scores):
            return np.array([[object()]], dtype=object)

        with self.assertRaisesRegex(
            ValueError,
            "score_to_cost must return real numeric cost matrices",
        ):
            solve_multisession_assignment_from_similarity(
                pairwise_scores,
                session_sizes=[1, 1],
                score_to_cost=score_to_cost,
            )

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_tracks_to_index_matrix_rejects_text_scalar_fill_value(self):
        for fill_value in self._text_scalars("-1"):
            with self.subTest(fill_value=fill_value):
                with self.assertRaisesRegex(
                    ValueError,
                    "fill_value must be a negative integer",
                ):
                    score_module.tracks_to_index_matrix(
                        [{0: 0}],
                        session_sizes=[1],
                        fill_value=fill_value,
                    )

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_tracks_to_index_matrix_rejects_temporal_scalar_fill_value(self):
        for fill_value in self._negative_temporal_scalars():
            with self.subTest(fill_value=fill_value):
                with self.assertRaisesRegex(
                    ValueError,
                    "fill_value must be a negative integer",
                ):
                    score_module.tracks_to_index_matrix(
                        [{0: 0}],
                        session_sizes=[1],
                        fill_value=fill_value,
                    )

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_tracks_to_index_matrix_rejects_duplicate_detections(self):
        with self.assertRaisesRegex(
            ValueError,
            "Each detection can only belong to a single track",
        ):
            score_module.tracks_to_index_matrix(
                [{0: 0}, {0: 0}],
                session_sizes=[1],
            )


if __name__ == "__main__":
    unittest.main()
