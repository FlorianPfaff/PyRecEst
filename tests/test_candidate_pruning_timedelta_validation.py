import unittest

import numpy as np

from pyrecest.utils import candidate_mask_from_costs


class TestCandidatePruningDurationValidation(unittest.TestCase):
    def test_cost_matrix_rejects_duration_arrays(self):
        matrix = np.array([[np.timedelta64(5, "s")]])
        with self.assertRaisesRegex(ValueError, "cost_matrix must be numeric"):
            candidate_mask_from_costs(matrix)


if __name__ == "__main__":
    unittest.main()
