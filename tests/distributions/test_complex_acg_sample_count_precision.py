from fractions import Fraction
import unittest

from pyrecest.distributions.hypersphere_subset.complex_angular_central_gaussian_distribution import (
    _validate_positive_sample_count,
)


class TestComplexAngularCentralGaussianSampleCountPrecision(unittest.TestCase):
    def test_rejects_fraction_rounded_to_integer_by_binary64(self):
        rounded_half_integer = Fraction(2**54 + 1, 2)
        self.assertTrue(float(rounded_half_integer).is_integer())

        with self.assertRaisesRegex(ValueError, "finite integer"):
            _validate_positive_sample_count(rounded_half_integer)

    def test_accepts_adjacent_exact_integer(self):
        exact_integer = Fraction(2**54 + 2, 2)

        self.assertEqual(
            _validate_positive_sample_count(exact_integer),
            2**53 + 1,
        )


if __name__ == "__main__":
    unittest.main()
