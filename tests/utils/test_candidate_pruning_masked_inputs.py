import unittest

import numpy as np
from pyrecest.utils.candidate_pruning import (
    CandidatePruningConfig,
    candidate_mask_from_costs,
)


class TestCandidatePruningMaskedInputs(unittest.TestCase):
    def test_masked_config_scalars_are_rejected(self):
        invalid_configs = (
            {"row_top_k": np.ma.array(2, mask=True)},
            {"column_top_k": np.ma.array(2, mask=True)},
            {"always_keep_finite": np.ma.array(True, mask=True)},
            {"probability_threshold": np.ma.array(0.5, mask=True)},
            {"max_cost": np.ma.array(4.0, mask=True)},
            {"max_cost_percentile": np.ma.array(50.0, mask=True)},
            {"large_cost": np.ma.array(100.0, mask=True)},
        )

        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    CandidatePruningConfig(**kwargs)

    def test_masked_cost_and_probability_entries_are_rejected(self):
        masked_costs = np.ma.array([[0.1, 1.0]], mask=[[True, False]])
        with self.assertRaisesRegex(ValueError, "cost_matrix must be numeric"):
            candidate_mask_from_costs(masked_costs)

        masked_probabilities = np.ma.array(
            [[0.9, 0.1]], mask=[[True, False]]
        )
        config = CandidatePruningConfig(probability_threshold=0.5)
        with self.assertRaisesRegex(ValueError, "probability_matrix must be numeric"):
            candidate_mask_from_costs(
                np.array([[1.0, 2.0]]),
                probability_matrix=masked_probabilities,
                config=config,
            )

    def test_unmasked_masked_arrays_remain_supported(self):
        costs = np.ma.array([[1.0, 2.0]], mask=False)
        config = CandidatePruningConfig(
            row_top_k=np.ma.array(1, mask=False)
        )

        np.testing.assert_array_equal(
            candidate_mask_from_costs(costs, config=config),
            np.array([[True, False]]),
        )


if __name__ == "__main__":
    unittest.main()
