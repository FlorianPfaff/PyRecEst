import unittest

from pyrecest.evaluation.get_distance_function import get_distance_function


class SymmetricDistanceFunctionValidationTest(unittest.TestCase):
    def test_rejects_invalid_symmetry_counts(self):
        for n_symm in (0, -1, 2.5, float("nan"), float("inf"), True, [2]):
            with self.subTest(n_symm=n_symm):
                with self.assertRaisesRegex(ValueError, "nSymm.*positive integer"):
                    get_distance_function("circle", nSymm=n_symm)

    def test_rejects_nonfinite_symmetry_offsets(self):
        for symmetry_offsets in ([0.0, float("nan")], [float("inf")], [-float("inf")]):
            with self.subTest(symmetry_offsets=symmetry_offsets):
                with self.assertRaisesRegex(ValueError, "symmetryOffsets.*finite"):
                    get_distance_function("circle", symmetryOffsets=symmetry_offsets)


if __name__ == "__main__":
    unittest.main()
