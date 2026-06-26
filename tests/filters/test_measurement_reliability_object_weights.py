import unittest

import pyrecest.backend
from pyrecest.backend import array
from pyrecest.filters import normalize_measurement_weights


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="object dtype regression coverage is specific to NumPy arrays",
)
class TestMeasurementReliabilityObjectWeights(unittest.TestCase):
    def test_object_weight_inputs_with_real_numeric_values_are_accepted(self):
        vector_weights = normalize_measurement_weights(
            array([1.0, 0.5, 0], dtype=object),
            3,
        )
        scalar_weights = normalize_measurement_weights(array(0.25, dtype=object), 2)

        self.assertEqual([float(value) for value in vector_weights], [1.0, 0.5, 0.0])
        self.assertEqual([float(value) for value in scalar_weights], [0.25, 0.25])

    def test_object_weight_inputs_still_reject_non_real_values(self):
        invalid_weights = (
            array([True, False], dtype=object),
            array(["0.5", "1.0"], dtype=object),
            array([complex(1.0, 0.0)], dtype=object),
        )

        for invalid_weight in invalid_weights:
            with self.subTest(weight=invalid_weight):
                with self.assertRaisesRegex(ValueError, "real numeric"):
                    normalize_measurement_weights(invalid_weight, 2)


if __name__ == "__main__":
    unittest.main()
