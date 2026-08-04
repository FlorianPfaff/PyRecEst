import unittest

import numpy as np
from pyrecest.utils.point_set_registration import AffineTransform


class TestAffineTransformComplexParameters(unittest.TestCase):
    def test_constructor_rejects_complex_matrix(self):
        invalid_matrices = (
            np.array([[1.0 + 0.0j]]),
            np.array([[1.0 + 2.0j]], dtype=object),
        )
        for matrix in invalid_matrices:
            with self.subTest(dtype=matrix.dtype):
                with self.assertRaisesRegex(ValueError, "matrix.*real-valued"):
                    AffineTransform(matrix, np.array([0.0]))

    def test_constructor_rejects_complex_offset(self):
        invalid_offsets = (
            np.array([1.0j]),
            np.array([1.0 + 2.0j], dtype=object),
        )
        for offset in invalid_offsets:
            with self.subTest(dtype=offset.dtype):
                with self.assertRaisesRegex(ValueError, "offset.*real-valued"):
                    AffineTransform(np.array([[1.0]]), offset)


if __name__ == "__main__":
    unittest.main()
