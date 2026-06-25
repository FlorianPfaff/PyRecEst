import unittest

import numpy as np

from pyrecest.utils import (
    CandidatePruningConfig,
    candidate_mask_from_costs,
    prune_pairwise_cost_matrix,
)


class TestCandidatePruningTemporalValidation(unittest.TestCase):
    def test_cost_matrix_rejects_datetime_and_timedelta_entries(self):
        invalid_matrices = (
            np.array([["2000-01-01"]], dtype="datetime64[D]"),
            np.array([[np.timedelta64(5, "s")]]),
            np.array([[np.datetime64("2000-01-01")]], dtype=object),
            np.array([[np.timedelta64(5, "s")]], dtype=object),
        )

        for function in (candidate_mask_from_costs, prune_pairwise_cost_matrix):
            for matrix in invalid_matrices:
                with self.subTest(function=function.__name__, dtype=str(matrix.dtype)):
                    with self.assertRaisesRegex(ValueError, "cost_matrix must be numeric"):
                        function(matrix)

    def test_probability_matrix_rejects_datetime_and_timedelta_entries(self):
        config = CandidatePruningConfig(probability_threshold=0.5)
        invalid_probability_matrices = (
            np.array([["2000-01-01"]], dtype="datetime64[D]"),
            np.array([[np.timedelta64(1, "s")]]),
            np.array([[np.datetime64("2000-01-01")]], dtype=object),
            np.array([[np.timedelta64(1, "s")]], dtype=object),
        )

        for probabilities in invalid_probability_matrices:
            with self.subTest(dtype=str(probabilities.dtype)):
                with self.assertRaisesRegex(ValueError, "probability_matrix must be numeric"):
                    candidate_mask_from_costs(
                        np.array([[1.0]]),
                        probability_matrix=probabilities,
                        config=config,
                    )


if __name__ == "__main__":
    unittest.main()
