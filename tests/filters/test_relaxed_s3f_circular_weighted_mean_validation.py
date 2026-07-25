import math
import unittest

from pyrecest.backend import array
from pyrecest.filters.relaxed_s3f_circular import circular_weighted_mean


class CircularWeightedMeanValidationTest(unittest.TestCase):
    def test_rejects_mismatched_angle_and_weight_counts(self):
        invalid_cases = [
            (array([0.7]), array([0.2, 0.8])),
            (array([0.0, 1.0]), array([1.0])),
        ]

        for angles, weights in invalid_cases:
            with self.subTest(angles=angles, weights=weights):
                with self.assertRaisesRegex(ValueError, "same number"):
                    circular_weighted_mean(angles, weights)

    def test_rejects_nonfinite_angles(self):
        for invalid_angle in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid_angle=invalid_angle):
                with self.assertRaisesRegex(ValueError, "angles must be finite"):
                    circular_weighted_mean(
                        array([0.0, invalid_angle]), array([0.5, 0.5])
                    )

    def test_rejects_undefined_antipodal_mean(self):
        with self.assertRaisesRegex(ValueError, "undefined"):
            circular_weighted_mean(
                array([0.0, math.pi]),
                array([0.5, 0.5]),
            )

    def test_valid_inputs_are_unchanged(self):
        mean_angle = circular_weighted_mean(
            array([0.0, 0.5 * math.pi]), array([0.5, 0.5])
        )

        self.assertAlmostEqual(mean_angle, 0.25 * math.pi)


if __name__ == "__main__":
    unittest.main()
