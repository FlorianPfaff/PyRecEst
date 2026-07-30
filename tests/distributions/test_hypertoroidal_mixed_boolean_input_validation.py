import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.distributions.hypertorus._input_validation import (
    as_hypertoroidal_points,
    as_shift_vector,
)


class TestHypertoroidalMixedBooleanInputValidation(unittest.TestCase):
    def test_rejects_mixed_boolean_shift_angles(self):
        for value in ([0.0, True], (np.bool_(False), 0.25)):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "boolean"):
                    as_shift_vector(value, 2)

    def test_rejects_mixed_boolean_evaluation_points(self):
        for value in ([[0.0, True]], [[np.bool_(False), 0.25]]):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "boolean"):
                    as_hypertoroidal_points(value, 2)

    def test_numeric_python_sequences_remain_valid(self):
        npt.assert_allclose(as_shift_vector([0.0, 1.0], 2), [0.0, 1.0])
        npt.assert_allclose(as_hypertoroidal_points([[0.0, 1.0]], 2), [[0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
