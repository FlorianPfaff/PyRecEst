"""Regression tests for score-native multi-session index matrices."""

import unittest

import numpy as np
from pyrecest.backend import (  # pylint: disable=no-name-in-module
    __backend_name__,
    array,
    array_equal,
)
from pyrecest.utils.multisession_assignment_score import tracks_to_index_matrix


class TestMultiSessionAssignmentIndexMatrix(unittest.TestCase):
    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_fill_value_must_not_collide_with_detection_indices(self):
        invalid_fill_values = (0, np.array(0), True, 1.5, np.array([-1]))

        for fill_value in invalid_fill_values:
            with self.subTest(fill_value=fill_value):
                with self.assertRaisesRegex(
                    ValueError,
                    "fill_value must be a negative integer",
                ):
                    tracks_to_index_matrix(
                        [{0: 0}],
                        session_sizes=[1, 0],
                        fill_value=fill_value,
                    )

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_negative_fill_value_preserves_missing_sessions(self):
        matrix = tracks_to_index_matrix(
            [{0: 0, 2: 1}],
            session_sizes=[1, 0, 2],
            fill_value=-1,
        )

        self.assertTrue(array_equal(matrix, array([[0, -1, 1]], dtype=int)))

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_fill_value_must_fit_in_int64(self):
        int64_min = int(np.iinfo(np.int64).min)

        for fill_value in (int64_min - 1, -(1 << 100)):
            with self.subTest(fill_value=fill_value):
                with self.assertRaisesRegex(ValueError, "fit in int64"):
                    tracks_to_index_matrix(
                        [{0: 0}],
                        session_sizes=[1, 0],
                        fill_value=fill_value,
                    )

        matrix = tracks_to_index_matrix(
            [{0: 0}],
            session_sizes=[1, 0],
            fill_value=int64_min,
        )
        self.assertEqual(int(matrix[0, 1]), int64_min)


if __name__ == "__main__":
    unittest.main()
