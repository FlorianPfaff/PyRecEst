import unittest
from fractions import Fraction

import numpy as np
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

    def test_rejects_temporal_sample_counts(self):
        temporal_counts = (
            np.timedelta64(2, "ns"),
            np.timedelta64(2, "us"),
            np.datetime64("1970-01-01T00:00:00.000000002", "ns"),
            np.asarray(np.timedelta64(2, "ns")),
            np.array(np.timedelta64(2, "ns"), dtype=object),
        )

        for count in temporal_counts:
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "integer"):
                    _validate_positive_sample_count(count)


if __name__ == "__main__":
    unittest.main()
