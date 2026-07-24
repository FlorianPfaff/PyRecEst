import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def _matmul_device_contract_code(target_module):
    return f"""
import torch
import pyrecest  # noqa: F401  # triggers backend-support compatibility patches
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch


def _non_cpu_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("meta")


target = {target_module}
device = _non_cpu_device()

right_matrix = torch.eye(2, device=device)
matrix_result = target.matmul(torch.eye(2), right_matrix)
assert matrix_result.device.type == device.type
assert tuple(matrix_result.shape) == (2, 2)
if device.type != "meta":
    assert torch.allclose(matrix_result.cpu(), torch.eye(2))

array_like_result = target.matmul([[1.0, 2.0]], torch.ones((2, 1), device=device))
assert array_like_result.device.type == device.type
assert tuple(array_like_result.shape) == (1, 1)
if device.type != "meta":
    assert torch.allclose(array_like_result.cpu(), torch.tensor([[3.0]]))

vector_result = target.matmul(torch.tensor([1.0, 2.0]), torch.ones(2, device=device))
assert vector_result.device.type == device.type
assert tuple(vector_result.shape) == ()
if device.type != "meta":
    assert torch.allclose(vector_result.cpu(), torch.tensor(3.0))

print("ok")
"""


def test_raw_pytorch_matmul_prefers_existing_non_cpu_device_after_import():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("numpy", _matmul_device_contract_code("raw_pytorch"))

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_public_pytorch_matmul_prefers_existing_non_cpu_device():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("pytorch", _matmul_device_contract_code("backend"))

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
