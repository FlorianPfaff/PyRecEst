import unittest

import numpy as np

from pyrecest.evaluation.get_distance_function import get_distance_function


class EuclideanMttDistanceTest(unittest.TestCase):
    def test_rejects_invalid_cutoff_distances(self):
        for cutoff_distance in (-1.0, float("nan"), float("inf"), True, [1.0]):
            with self.subTest(cutoff_distance=cutoff_distance):
                with self.assertRaisesRegex(ValueError, "cutoff_distance.*finite"):
                    get_distance_function(
                        "euclidean_mtt",
                        {"cutoff_distance": cutoff_distance},
                    )

    def test_unmatched_targets_use_nonnegative_cutoff_distance(self):
        distance = get_distance_function(
            "euclidean_mtt",
            {"cutoff_distance": 2.5},
        )

        self.assertEqual(distance(np.empty((0, 2)), np.array([[1.0, 2.0]])), 2.5)


if __name__ == "__main__":
    unittest.main()
