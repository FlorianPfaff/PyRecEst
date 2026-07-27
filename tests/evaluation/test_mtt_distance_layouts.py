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
        self.assertEqual(self.distance(np.array([]), np.zeros((3, 2))), 30.0)

    def test_preserves_ambiguous_dimension_first_preference(self):
        first = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        second = np.zeros((2, 5))

        self.assertEqual(self.distance(first, second), 10.0)


if __name__ == "__main__":
    unittest.main()
