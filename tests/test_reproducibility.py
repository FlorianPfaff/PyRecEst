import unittest

import numpy as np
from pyrecest.reproducibility import _normalize_seed


class ReproducibilityValidationTest(unittest.TestCase):
    def test_normalize_seed_rejects_text_values(self):
        for value in ("1", np.array("1")):
            with self.subTest(value=repr(value)):
                with self.assertRaisesRegex(
                    ValueError,
                    "seed must be a non-negative integer or None",
                ):
                    _normalize_seed(value)

    def test_normalize_seed_preserves_numeric_scalar_support(self):
        self.assertIsNone(_normalize_seed(None))
        self.assertEqual(_normalize_seed(1), 1)
        self.assertEqual(_normalize_seed(2.0), 2)
        self.assertEqual(_normalize_seed(np.array(3)), 3)
        self.assertEqual(_normalize_seed(np.array(4.0)), 4)


if __name__ == "__main__":
    unittest.main()
