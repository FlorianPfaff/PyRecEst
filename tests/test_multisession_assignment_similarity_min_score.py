"""Regression tests for score-native multi-session min_score validation."""

import unittest

import numpy as np
from pyrecest.backend import __backend_name__, array
from pyrecest.utils import solve_multisession_assignment_from_similarity


class TestMultiSessionAssignmentSimilarityMinScore(unittest.TestCase):
    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_accepts_scalar_float_min_score(self):
        result = solve_multisession_assignment_from_similarity(
            [array([[0.25]], dtype=float)],
            min_score=np.array(0.2),
            start_cost=1.0,
            end_cost=1.0,
        )

        self.assertEqual(result.tracks, [{0: 0, 1: 0}])

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_rejects_invalid_min_score_scalars(self):
        pairwise_scores = [array([[0.25]], dtype=float)]
        invalid_min_scores = (True, np.nan, np.inf, -np.inf, np.array([0.2]))

        for min_score in invalid_min_scores:
            with self.subTest(min_score=min_score):
                with self.assertRaisesRegex(
                    ValueError,
                    "min_score must be a finite scalar",
                ):
                    solve_multisession_assignment_from_similarity(
                        pairwise_scores,
                        min_score=min_score,
                    )


if __name__ == "__main__":
    unittest.main()
