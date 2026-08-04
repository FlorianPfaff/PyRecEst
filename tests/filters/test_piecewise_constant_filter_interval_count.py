import unittest

import numpy as np

from pyrecest.filters import PiecewiseConstantFilter


class PiecewiseConstantFilterIntervalCountTest(unittest.TestCase):
    def test_accepts_exact_numpy_integer_scalars(self):
        for count in (
            np.int64(3),
            np.array(3, dtype=np.int64),
            np.ma.array(3, mask=False, dtype=np.int64),
        ):
            with self.subTest(count=repr(count)):
                filter_obj = PiecewiseConstantFilter(count)
                self.assertEqual(len(filter_obj.filter_state.w), 3)

    def test_rejects_non_integer_or_masked_counts(self):
        invalid_counts = (
            True,
            np.bool_(True),
            3.0,
            np.array(3.0),
            np.array([3], dtype=np.int64),
            np.ma.array(3, mask=True, dtype=np.int64),
            0,
            -1,
        )
        for count in invalid_counts:
            with self.subTest(count=repr(count)):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    PiecewiseConstantFilter(count)


if __name__ == "__main__":
    unittest.main()
