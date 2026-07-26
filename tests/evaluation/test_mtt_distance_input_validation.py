import unittest

import numpy as np
from pyrecest.evaluation.get_distance_function import get_distance_function


class EuclideanMttDistanceInputValidationTest(unittest.TestCase):
    def test_rejects_text_cutoff_distances(self):
        for cutoff_distance in ("2.5", np.array("2.5", dtype=object)):
            with self.subTest(cutoff_distance=cutoff_distance):
                with self.assertRaisesRegex(ValueError, "cutoff_distance.*finite"):
                    get_distance_function(
                        "euclidean_mtt",
                        {"cutoff_distance": cutoff_distance},
                    )

    def test_rejects_invalid_state_values(self):
        distance = get_distance_function(
            "euclidean_mtt",
            {"cutoff_distance": 2.5},
        )
        valid_state = np.array([[0.0, 0.0]])
        invalid_states = (
            np.array([[1.0 + 2.0j, 0.0]]),
            [[True, 0.0]],
            [["1.0", 0.0]],
            [[np.nan, 0.0]],
            [[np.inf, 0.0]],
        )

        for state in invalid_states:
            with self.subTest(state=state, argument="x1"):
                with self.assertRaisesRegex(ValueError, "x1.*finite real"):
                    distance(state, valid_state)
            with self.subTest(state=state, argument="x2"):
                with self.assertRaisesRegex(ValueError, "x2.*finite real"):
                    distance(valid_state, state)

    def test_rejects_invalid_state_ranks(self):
        distance = get_distance_function(
            "euclidean_mtt",
            {"cutoff_distance": 2.5},
        )
        valid_state = np.array([[0.0, 0.0]])
        invalid_states = (
            1.0,
            np.zeros((2, 1, 1)),
            np.empty((0, 1, 2)),
        )

        for state in invalid_states:
            with self.subTest(state_shape=np.asarray(state).shape, argument="x1"):
                with self.assertRaisesRegex(ValueError, "x1.*one- or two-dimensional"):
                    distance(state, valid_state)
            with self.subTest(state_shape=np.asarray(state).shape, argument="x2"):
                with self.assertRaisesRegex(ValueError, "x2.*one- or two-dimensional"):
                    distance(valid_state, state)

    def test_valid_empty_target_behavior_is_preserved(self):
        distance = get_distance_function(
            "euclidean_mtt",
            {"cutoff_distance": 2.5},
        )

        self.assertEqual(distance(np.empty((0, 2)), np.array([[1.0, 2.0]])), 2.5)


if __name__ == "__main__":
    unittest.main()
