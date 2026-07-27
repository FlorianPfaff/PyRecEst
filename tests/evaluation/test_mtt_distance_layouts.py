import unittest

import numpy as np
from pyrecest.evaluation.get_distance_function import get_distance_function


class EuclideanMttDistanceLayoutTest(unittest.TestCase):
    def setUp(self):
        self.distance = get_distance_function(
            "euclidean_mtt",
            {"cutoff_distance": 10.0},
        )
        self.first = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        self.second = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0, 0.0, 0.0],
                [20.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

    def test_preserves_row_major_high_dimensional_target_sets(self):
        self.assertEqual(self.distance(self.first, self.second), 10.0)

    def test_preserves_dimension_first_target_sets(self):
        self.assertEqual(self.distance(self.first.T, self.second.T), 10.0)

    def test_supports_dimension_first_empty_target_sets(self):
        self.assertEqual(self.distance(np.empty((5, 0)), self.second.T), 30.0)

    def test_preserves_dimensionless_empty_target_sets(self):
        self.assertEqual(self.distance(np.array([]), self.second), 30.0)


if __name__ == "__main__":
    unittest.main()
