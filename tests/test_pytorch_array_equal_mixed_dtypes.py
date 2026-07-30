"""Regression tests for mixed-dtype PyTorch array equality."""

import unittest
from typing import Any

import numpy as np

pytorch_backend: Any
try:
    import pyrecest  # noqa: F401  # Apply runtime backend compatibility patches.
    from pyrecest._backend import pytorch as pytorch_backend
except ModuleNotFoundError:
    pytorch_backend = None


@unittest.skipIf(pytorch_backend is None, "PyTorch is not installed")
class TestPytorchArrayEqualMixedDtypes(unittest.TestCase):
    def test_mixed_int64_float32_comparison_matches_numpy(self):
        integer_value = 2**24 + 1
        rounded_float_value = float(2**24)
        integer_tensor = pytorch_backend.array(
            [integer_value], dtype=pytorch_backend.int64
        )
        float_tensor = pytorch_backend.array(
            [rounded_float_value], dtype=pytorch_backend.float32
        )

        expected = np.array_equal(
            np.array([integer_value], dtype=np.int64),
            np.array([rounded_float_value], dtype=np.float32),
        )

        self.assertFalse(expected)
        self.assertEqual(
            pytorch_backend.array_equal(integer_tensor, float_tensor), expected
        )
        self.assertEqual(
            pytorch_backend.array_equal(integer_tensor, float_tensor, equal_nan=True),
            expected,
        )

    def test_exactly_equal_mixed_values_remain_equal(self):
        integer_value = 2**24
        integer_tensor = pytorch_backend.array(
            [integer_value], dtype=pytorch_backend.int64
        )
        float_tensor = pytorch_backend.array(
            [float(integer_value)], dtype=pytorch_backend.float32
        )

        self.assertTrue(pytorch_backend.array_equal(integer_tensor, float_tensor))


if __name__ == "__main__":
    unittest.main()
