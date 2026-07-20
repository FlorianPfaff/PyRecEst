import importlib.util

import pytest

from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def _linspace_device_contract_code(target_module):
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
endpoint_pairs = [
    (torch.tensor(0.0), torch.tensor(1.0, device=device)),
    (torch.tensor(0.0, device=device), torch.tensor(1.0)),
]

for start, stop in endpoint_pairs:
    result = target.linspace(start, stop, num=3)

    assert result.device.type == device.type
    assert result.shape == (3,)
    if device.type != "meta":
        expected = torch.tensor([0.0, 0.5, 1.0], device=device)
        assert torch.allclose(result, expected)

print("ok")
"""


def test_raw_pytorch_linspace_prefers_existing_non_cpu_endpoint_device():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "numpy", _linspace_device_contract_code("raw_pytorch")
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_public_pytorch_linspace_prefers_existing_non_cpu_endpoint_device():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch", _linspace_device_contract_code("backend")
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
