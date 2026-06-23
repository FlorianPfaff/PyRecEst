import re
import unittest

import numpy as np

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.utils import LogisticPairwiseAssociationModel


class TestLogisticAssociationScalarValidation(unittest.TestCase):
    def test_constructor_rejects_invalid_scalar_controls(self):
        cases = (
            (
                "l2_regularization",
                "l2_regularization must be non-negative",
                (-1.0, np.nan, np.inf, True, np.array([0.0])),
            ),
            (
                "max_iterations",
                "max_iterations must be positive",
                (0, 1.5, np.nan, np.inf, True, np.array([2])),
            ),
            (
                "tolerance",
                "tolerance must be positive",
                (0.0, -1.0, np.nan, np.inf, True, np.array([1.0e-8])),
            ),
            (
                "probability_clip",
                "probability_clip must lie in (0, 0.5)",
                (0.0, 0.5, np.nan, np.inf, True, np.array([0.1])),
            ),
        )

        for field_name, message, values in cases:
            for value in values:
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaisesRegex(ValueError, re.escape(message)):
                        LogisticPairwiseAssociationModel(**{field_name: value})

    def test_constructor_accepts_integer_like_max_iterations(self):
        model = LogisticPairwiseAssociationModel(max_iterations=np.array(3.0))

        self.assertEqual(model.max_iterations, 3)

    def test_class_weight_rejects_boolean_and_non_scalar_weights(self):
        features = array([[0.0], [1.0]])
        labels = array([0, 1])

        for invalid_weight in (True, np.array([1.0])):
            with self.subTest(invalid_weight=invalid_weight):
                model = LogisticPairwiseAssociationModel(
                    class_weight={0: invalid_weight, 1: 1.0}
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "class weights must be finite and positive",
                ):
                    model.fit(features, labels)


if __name__ == "__main__":
    unittest.main()
