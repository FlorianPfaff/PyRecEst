import unittest
import warnings

import numpy as np
from pyrecest.distributions.cart_prod.se2_pwn_distribution import SE2PWNDistribution


class TestSE2PWNFromSamplesValidation(unittest.TestCase):
    def test_rejects_malformed_or_too_short_sample_sets(self):
        invalid_samples = (
            np.array([0.0, 1.0, 2.0]),
            np.zeros((2, 2)),
            np.zeros((2, 4)),
        )
        for samples in invalid_samples:
            with self.subTest(shape=samples.shape):
                with self.assertRaisesRegex(ValueError, r"shape \(n, 3\)"):
                    SE2PWNDistribution.from_samples(samples)

        with self.assertRaisesRegex(ValueError, "at least two observations"):
            SE2PWNDistribution.from_samples(np.zeros((1, 3)))

    def test_rejects_non_real_or_missing_sample_values(self):
        masked = np.ma.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            mask=[[False, False, False], [True, False, False]],
        )
        invalid_samples = (
            np.array([[0.0, 0.0, 0.0], [np.nan, 1.0, 1.0]]),
            np.array([[0.0, 0.0, 0.0], [np.inf, 1.0, 1.0]]),
            np.array([[False, False, False], [True, True, True]]),
            np.array([[0.0, 0.0, 0.0], [1.0j, 1.0, 1.0]]),
            np.array([["0", "0", "0"], ["1", "1", "1"]]),
            masked,
        )
        for samples in invalid_samples:
            with self.subTest(dtype=getattr(samples, "dtype", None)):
                with self.assertRaisesRegex(ValueError, "finite real numeric"):
                    SE2PWNDistribution.from_samples(samples)

    def test_rejects_nested_mask_without_numpy_conversion_warning(self):
        samples = [[0.0, 0.0, 0.0], [np.ma.masked, 1.0, 1.0]]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaisesRegex(ValueError, "finite real numeric"):
                SE2PWNDistribution.from_samples(samples)

    def test_rejects_unidentifiable_zero_resultant_angle_samples(self):
        samples = np.column_stack(
            [
                np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]),
                np.arange(4.0),
                np.array([0.0, 1.0, 4.0, 9.0]),
            ]
        )

        with self.assertRaisesRegex(ValueError, "first trigonometric moment"):
            SE2PWNDistribution.from_samples(samples)


if __name__ == "__main__":
    unittest.main()
