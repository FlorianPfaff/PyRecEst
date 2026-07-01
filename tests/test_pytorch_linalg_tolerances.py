import unittest
from typing import Any

import numpy as np

pytorch_backend: Any
try:
    from pyrecest._backend import pytorch as pytorch_backend
except ModuleNotFoundError:
    pytorch_backend = None


@unittest.skipIf(pytorch_backend is None, "PyTorch is not installed")
class TestPytorchLinalgToleranceContract(unittest.TestCase):
    def test_matrix_rank_accepts_numpy_scalar_array_tolerances(self):
        value = pytorch_backend.diag(pytorch_backend.array([1.0, 1e-5]))

        self.assertEqual(
            int(pytorch_backend.linalg.matrix_rank(value, tol=np.array(1e-4))),
            1,
        )
        self.assertEqual(
            int(pytorch_backend.linalg.matrix_rank(value, rtol=np.array(1e-4))),
            1,
        )

    def test_pinv_accepts_numpy_scalar_array_tolerances(self):
        value = pytorch_backend.diag(
            pytorch_backend.array([1.0, 1e-5], dtype=pytorch_backend.float64)
        )
        expected = pytorch_backend.diag(
            pytorch_backend.array([1.0, 0.0], dtype=pytorch_backend.float64)
        )

        for keyword in ("rcond", "rtol"):
            with self.subTest(keyword=keyword):
                result = pytorch_backend.linalg.pinv(
                    value,
                    **{keyword: np.array(1e-4)},
                )

                self.assertEqual(result.dtype, pytorch_backend.float64)
                self.assertTrue(pytorch_backend.allclose(result, expected))


if __name__ == "__main__":
    unittest.main()
