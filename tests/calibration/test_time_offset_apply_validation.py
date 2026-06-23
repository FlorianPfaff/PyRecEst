import unittest

import numpy as np
from pyrecest.calibration.time_offset import apply_time_offset


class ApplyTimeOffsetValidationTest(unittest.TestCase):
    def test_apply_time_offset_rejects_invalid_offsets(self):
        times = np.array([0.0, 1.0])

        for offset_s in (np.nan, np.inf, -np.inf, True, np.array([0.5])):
            with self.subTest(offset_s=offset_s):
                with self.assertRaisesRegex(ValueError, "offset_s"):
                    apply_time_offset(times, offset_s)

    def test_apply_time_offset_accepts_scalar_numpy_offset(self):
        shifted = apply_time_offset(np.array([0.0, 1.0]), np.array(0.25))

        np.testing.assert_allclose(shifted, np.array([0.25, 1.25]))


if __name__ == "__main__":
    unittest.main()
