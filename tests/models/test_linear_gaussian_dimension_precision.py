import unittest
from fractions import Fraction

import numpy as np
import pyrecest.backend
from pyrecest.models.linear_gaussian import _as_positive_integer


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="object dtype regression coverage is specific to NumPy arrays",
)
class LinearGaussianDimensionPrecisionTest(unittest.TestCase):
    def test_rejects_fractional_dimension_hidden_by_float_rounding(self):
        fractional_dim = np.array(Fraction(2**54 + 1, 2), dtype=object)

        with self.assertRaisesRegex(ValueError, "dim must be a positive integer"):
            _as_positive_integer(fractional_dim, "dim")

    def test_preserves_exact_integral_object_scalar(self):
        exact_dim = np.array(Fraction(6, 3), dtype=object)

        self.assertEqual(_as_positive_integer(exact_dim, "dim"), 2)


if __name__ == "__main__":
    unittest.main()
