import unittest

import numpy as np
from pyrecest.calibration import apply_time_offset, make_offset_grid


class TimeOffsetRealNumericValidationTest(unittest.TestCase):
    def test_apply_time_offset_rejects_numpy_complex_scalar_inside_object_array(self):
        times = np.array([np.complex128(complex(1.0, 2.0))], dtype=object)

        with self.assertRaisesRegex(
            ValueError,
            "times_s must contain real numeric values",
        ):
            apply_time_offset(times, 0.0)

    def test_make_offset_grid_rejects_numpy_complex_object_scalar(self):
        min_s = np.array(np.complex128(complex(0.0, 1.0)), dtype=object)

        with self.assertRaisesRegex(ValueError, "min_s must be a finite scalar"):
            make_offset_grid(min_s, 1.0, 1.0)


if __name__ == "__main__":
    unittest.main()
