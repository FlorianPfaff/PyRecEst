import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def _scalar_contract_code(target_module):
    return f"""
import torch
import pyrecest  # noqa: F401  # triggers backend-support compatibility patches
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch


target = {target_module}

scalar_result = target.one_hot(2, 4)
assert scalar_result.dtype == torch.uint8
assert tuple(scalar_result.shape) == (4,)
assert torch.equal(scalar_result, torch.tensor([0, 0, 1, 0], dtype=torch.uint8))

batched_result = target.one_hot([0, 2], 4)
assert batched_result.dtype == torch.uint8
assert tuple(batched_result.shape) == (2, 4)
assert torch.equal(
    batched_result,
    torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=torch.uint8),
)

for bad_num_classes in (True, torch.tensor(True)):
    try:
        target.one_hot([0], bad_num_classes)
    except TypeError as exc:
        assert "num_classes must be an integer" in str(exc)
    else:
        raise AssertionError("boolean num_classes was accepted")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("meta")
device_result = target.one_hot(torch.tensor(2, device=device), 4)
assert device_result.device.type == device.type
assert tuple(device_result.shape) == (4,)

print("ok")
"""


def test_raw_pytorch_one_hot_accepts_scalar_label_after_import():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("numpy", _scalar_contract_code("raw_pytorch"))

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_public_pytorch_one_hot_accepts_scalar_label():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("pytorch", _scalar_contract_code("backend"))

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
