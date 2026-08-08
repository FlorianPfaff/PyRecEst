import unittest

import numpy as np
from pyrecest.filters import so3_right_multiplication_grid_transition


class TestSO3GridTransitionComplexInputs(unittest.TestCase):
    def setUp(self):
        self.grid = np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )

    def test_rejects_complex_grid_without_discarding_imaginary_parts(self):
        complex_grid = self.grid.astype(np.complex128)
        complex_grid[0, 0] = 0.25j

        with self.assertRaisesRegex(ValueError, "grid quaternions.*real"):
            so3_right_multiplication_grid_transition(
                complex_grid,
                np.zeros(3),
                1.0,
            )

    def test_rejects_complex_tangent_increment(self):
        complex_increment = np.array([0.1 + 0.2j, 0.0, 0.0])

        with self.assertRaisesRegex(ValueError, "orientation_increment.*real"):
            so3_right_multiplication_grid_transition(
                self.grid,
                complex_increment,
                1.0,
            )

    def test_rejects_complex_quaternion_increment(self):
        complex_increment = np.array([0.25j, 0.0, 0.0, 1.0])

        with self.assertRaisesRegex(ValueError, "orientation_increment.*real"):
            so3_right_multiplication_grid_transition(
                self.grid,
                complex_increment,
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
