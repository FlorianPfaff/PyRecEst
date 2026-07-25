import unittest
from typing import Any

pytorch_backend: Any
try:
    from pyrecest._backend import pytorch as pytorch_backend
except ModuleNotFoundError:
    pytorch_backend = None


@unittest.skipIf(pytorch_backend is None, "PyTorch is not installed")
class TestPytorchBackendCloseMissingValues(unittest.TestCase):
    def test_allclose_accepts_equal_nan_for_raw_backend(self):
        left = pytorch_backend.array([1.0, float("nan")])
        right = pytorch_backend.array([1.0, float("nan")])

        self.assertFalse(bool(pytorch_backend.allclose(left, right)))
        self.assertTrue(bool(pytorch_backend.allclose(left, right, equal_nan=True)))

    def test_isclose_accepts_equal_nan_for_raw_backend(self):
        left = pytorch_backend.array([1.0, float("nan"), 3.0])
        right = pytorch_backend.array([1.0, float("nan"), 4.0])

        self.assertEqual(
            pytorch_backend.isclose(left, right).tolist(), [True, False, False]
        )
        self.assertEqual(
            pytorch_backend.isclose(left, right, equal_nan=True).tolist(),
            [True, True, False],
        )


if __name__ == "__main__":
    unittest.main()
