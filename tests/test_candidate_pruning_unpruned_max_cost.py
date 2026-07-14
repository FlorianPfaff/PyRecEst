import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.utils import CandidatePruningConfig, prune_pairwise_cost_matrix


class TestCandidatePruningUnprunedMaxCost(unittest.TestCase):
    def test_unpruned_max_float_does_not_require_larger_finite_penalty(self):
        costs = np.array([[np.finfo(float).max]])

        pruned = prune_pairwise_cost_matrix(
            costs,
            config=CandidatePruningConfig(always_keep_finite=True),
        )

        npt.assert_array_equal(pruned, costs)
        self.assertTrue(np.all(np.isfinite(pruned)))


if __name__ == "__main__":
    unittest.main()
