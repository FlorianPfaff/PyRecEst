import unittest

import numpy as np
from pyrecest.evaluation.get_distance_function import get_distance_function


class SymmetricDistanceNumericValidationTest(unittest.TestCase):
    def test_rejects_text_symmetry_count(self):
        with self.assertRaisesRegex(ValueError, "nSymm.*positive integer"):
            get_distance_function("circle", nSymm="2")

    def test_rejects_text_symmetry_offsets(self):
        with self.assertRaisesRegex(ValueError, "symmetryOffsets.*numeric"):
            get_distance_function("circle", symmetryOffsets=["0.0"])

    def test_rejects_boolean_symmetry_offsets(self):
        with self.assertRaisesRegex(ValueError, "symmetryOffsets.*numeric"):
            get_distance_function(
                "circle", symmetryOffsets=np.array([True], dtype=object)
            )


if __name__ == "__main__":
    unittest.main()
