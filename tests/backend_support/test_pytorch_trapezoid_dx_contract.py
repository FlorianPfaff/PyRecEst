import unittest
from typing import Any

from pyrecest.backend_support._pytorch_trapezoid_numpy_contract import (
    patch_pytorch_trapezoid_numpy_contract,
)

pytorch_backend: Any
try:
    from pyrecest._backend import pytorch as pytorch_backend
except ModuleNotFoundError:
    pytorch_backend = None


@unittest.skipIf(pytorch_backend is None, "PyTorch is not installed")
class TestPytorchTrapezoidDxContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        patch_pytorch_trapezoid_numpy_contract()

    def test_accepts_array_like_dx(self):
        result = pytorch_backend.trapezoid(
            [[1.0, 2.0, 4.0], [2.0, 5.0, 8.0]],
            dx=[0.25, 1.5],
            axis=-1,
        )

        expected = pytorch_backend.array([4.875, 10.625])
        self.assertTrue(pytorch_backend.allclose(result, expected))

    def test_accepts_complex_dx(self):
        result = pytorch_backend.trapezoid([1.0, 2.0, 3.0], dx=1.0 + 2.0j)

        self.assertEqual(complex(result.item()), 4.0 + 8.0j)

    def test_preserves_real_scalar_dx_behavior(self):
        result = pytorch_backend.trapezoid([1.0, 2.0, 5.0], dx=0.5)

        self.assertAlmostEqual(float(result), 2.5)


if __name__ == "__main__":
    unittest.main()
