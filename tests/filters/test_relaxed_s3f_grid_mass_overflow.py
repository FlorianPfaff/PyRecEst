import unittest

import numpy as np
from pyrecest.backend import allclose, array
from pyrecest.filters.relaxed_s3f_circular import (
    circular_weighted_mean,
    grid_probability_masses,
)


class RelaxedS3FGridMassOverflowTest(unittest.TestCase):
    def test_large_finite_grid_values_normalize_without_overflow(self):
        max_value = np.finfo(float).max

        masses = grid_probability_masses(array([max_value, max_value / 2.0]))

        self.assertTrue(bool(allclose(masses, array([2.0 / 3.0, 1.0 / 3.0]))))
        self.assertAlmostEqual(float(masses.sum()), 1.0)

    def test_circular_mean_is_invariant_to_large_common_weight_scale(self):
        angles = array([0.0, np.pi / 2.0])
        max_value = np.finfo(float).max

        ordinary = circular_weighted_mean(angles, array([2.0, 1.0]))
        scaled = circular_weighted_mean(
            angles,
            array([max_value, max_value / 2.0]),
        )

        self.assertAlmostEqual(scaled, ordinary)


if __name__ == "__main__":
    unittest.main()
