import unittest

import pyrecest.backend
from pyrecest.backend import array
from pyrecest.filters import (
    normalize_active_measurement_mask,
    normalize_measurement_weights,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="measurement reliability tests currently use backend array shape checks",
)
class TestMeasurementReliabilityShapes(unittest.TestCase):
    def test_weight_matrices_are_rejected_instead_of_flattened(self):
        matrix_weights = (
            array([[1.0, 0.5]]),
            array([[1.0], [0.5]]),
        )

        for weights in matrix_weights:
            with self.subTest(shape=weights.shape):
                with self.assertRaisesRegex(ValueError, "one-dimensional"):
                    normalize_measurement_weights(weights, 2)

    def test_mask_matrices_are_rejected_instead_of_flattened(self):
        matrix_masks = (
            array([[True, False]]),
            array([[True], [False]]),
        )

        for mask in matrix_masks:
            with self.subTest(shape=mask.shape):
                with self.assertRaisesRegex(ValueError, "one-dimensional"):
                    normalize_active_measurement_mask(mask, 2)


if __name__ == "__main__":
    unittest.main()
