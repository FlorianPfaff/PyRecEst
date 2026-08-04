import math
import unittest

import numpy as np
from pyrecest.backend import array, float64
from pyrecest.utils._point_set_registration_common import compute_rmse


class TestPointSetRegistrationRmseStability(unittest.TestCase):
    def test_rmse_preserves_maximum_finite_cost(self):
        largest = np.finfo(np.float64).max

        rmse = compute_rmse(array([largest], dtype=float64))

        self.assertTrue(math.isfinite(rmse))
        self.assertEqual(rmse, largest)

    def test_rmse_preserves_zero_and_empty_contract(self):
        self.assertEqual(compute_rmse(array([0.0], dtype=float64)), 0.0)
        self.assertTrue(math.isinf(compute_rmse(array([], dtype=float64))))


if __name__ == "__main__":
    unittest.main()
