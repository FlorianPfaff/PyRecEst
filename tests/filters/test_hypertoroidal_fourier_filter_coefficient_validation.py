import unittest

import numpy as np

import pyrecest.backend
from pyrecest.filters.hypertoroidal_fourier_filter import HypertoroidalFourierFilter


@unittest.skipIf(
    pyrecest.backend.__backend_name__ in ("jax", "pytorch"),
    reason="HypertoroidalFourierFilter is not supported on this backend",
)
class TestHypertoroidalFourierFilterCoefficientValidation(unittest.TestCase):
    def test_accepts_numpy_integer_scalar(self):
        fourier_filter = HypertoroidalFourierFilter(np.int64(11))

        self.assertEqual(fourier_filter.filter_state.coeff_mat.shape, (11,))

    def test_rejects_values_that_would_be_silently_reinterpreted(self):
        invalid_values = (
            True,
            "11",
            (),
            (11.5,),
            (11.0,),
        )

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    HypertoroidalFourierFilter(value)

    def test_rejects_nonpositive_or_even_coefficient_counts(self):
        invalid_values = (0, -1, 10, (11, 0), (11, 12))

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "n_coefficients"):
                    HypertoroidalFourierFilter(value)
