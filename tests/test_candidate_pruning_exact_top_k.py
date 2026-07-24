from fractions import Fraction
import unittest

from pyrecest.utils import CandidatePruningConfig


class TestCandidatePruningExactTopK(unittest.TestCase):
    def test_rejects_fraction_rounded_to_integer_by_binary64(self):
        fractional_top_k = Fraction(2**54 + 1, 2)

        for field_name in ("row_top_k", "column_top_k"):
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(
                    ValueError,
                    f"{field_name} must be a positive integer or None",
                ):
                    CandidatePruningConfig(**{field_name: fractional_top_k})

    def test_accepts_exact_rational_integer(self):
        config = CandidatePruningConfig(row_top_k=Fraction(4, 2))

        self.assertEqual(config.row_top_k, 2)


if __name__ == "__main__":
    unittest.main()
