import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def _require_torch():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")


def test_top_level_copy_tracks_patched_pytorch_backend_copy():
    _require_torch()

    code = """
import torch

import pyrecest
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch

assert getattr(backend, "__backend_name__", None) == "pytorch"
assert pyrecest.copy is backend.copy
assert backend.copy is raw_pytorch.copy

copied = pyrecest.copy([1.0, 2.0])
assert torch.is_tensor(copied), type(copied)
assert copied.dtype == backend.get_default_dtype()
assert backend.to_numpy(copied).tolist() == [1.0, 2.0]

source = torch.tensor([1.0, 2.0])
copied_tensor = pyrecest.copy(source)
assert torch.is_tensor(copied_tensor)
assert copied_tensor is not source
assert backend.to_numpy(copied_tensor).tolist() == [1.0, 2.0]
print("ok")
"""
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
