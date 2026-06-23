import unittest

import numpy as np
from pyrecest.calibration.bias import make_bias_training_examples
from pyrecest.calibration.time_offset import interpolate_reference_values


class MaxTimeDeltaValidationTest(unittest.TestCase):
    def test_interpolation_rejects_boolean_and_non_scalar_max_time_delta(self):
        invalid_values = (True, np.array([1.0]))

        for max_time_delta_s in invalid_values:
            with self.subTest(max_time_delta_s=max_time_delta_s):
                with self.assertRaisesRegex(
                    ValueError,
                    "max_time_delta_s must be nonnegative",
                ):
                    interpolate_reference_values(
                        np.array([0.0, 1.0]),
                        np.array([[0.0], [1.0]]),
                        np.array([0.5]),
                        max_time_delta_s=max_time_delta_s,
                    )

    def test_bias_examples_reject_boolean_and_non_scalar_max_time_delta(self):
        invalid_values = (True, np.array([1.0]))

        for max_time_delta_s in invalid_values:
            with self.subTest(max_time_delta_s=max_time_delta_s):
                with self.assertRaisesRegex(
                    ValueError,
                    "max_time_delta_s must be nonnegative",
                ):
                    make_bias_training_examples(
                        np.array([0.0]),
                        np.array([[1.0]]),
                        np.array([0.0]),
                        np.array([[0.0]]),
                        max_time_delta_s=max_time_delta_s,
                    )


if __name__ == "__main__":
    unittest.main()
