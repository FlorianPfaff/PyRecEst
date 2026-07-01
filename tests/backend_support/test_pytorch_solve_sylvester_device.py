import unittest
from typing import Any

pytorch_backend: Any
torch: Any
try:
    import torch
    from pyrecest._backend import pytorch as pytorch_backend
except ModuleNotFoundError:
    torch = None
    pytorch_backend = None


@unittest.skipIf(pytorch_backend is None, "PyTorch is not installed")
class TestPytorchSolveSylvesterDevice(unittest.TestCase):
    def test_solve_sylvester_keeps_common_dtype_for_mixed_array_like_inputs(self):
        a = pytorch_backend.array(
            [[2.0, 0.0], [0.0, 3.0]], dtype=pytorch_backend.float32
        )
        b = [[2.0, 0.0], [0.0, 3.0]]
        q = pytorch_backend.array(
            [[8.0, 10.0], [10.0, 12.0]], dtype=pytorch_backend.float64
        )

        result = pytorch_backend.linalg.solve_sylvester(a, b, q)

        expected = pytorch_backend.array(
            [[2.0, 2.0], [2.0, 2.0]], dtype=pytorch_backend.float64
        )
        self.assertEqual(result.dtype, pytorch_backend.float64)
        self.assertTrue(pytorch_backend.allclose(result, expected))

    @unittest.skipIf(
        torch is None or not torch.cuda.is_available(), "CUDA is not available"
    )
    def test_solve_sylvester_aligns_mixed_tensor_devices(self):
        a = torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=torch.float32, device="cuda")
        b = torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=torch.float32)
        q = torch.tensor([[8.0, 10.0], [10.0, 12.0]], dtype=torch.float64)

        result = pytorch_backend.linalg.solve_sylvester(a, b, q)

        expected = torch.tensor(
            [[2.0, 2.0], [2.0, 2.0]], dtype=torch.float64, device="cuda"
        )
        self.assertEqual(result.device.type, "cuda")
        self.assertEqual(result.dtype, torch.float64)
        self.assertTrue(torch.allclose(result, expected))


if __name__ == "__main__":
    unittest.main()
