import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def test_public_pytorch_asarray_resolves_torch_dtype_alias_strings():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = """
import pyrecest.backend as backend

assert backend.asarray([1.0], dtype="torch.float").dtype == backend.float32
assert backend.asarray([1.0], dtype="torch.double").dtype == backend.float64
assert backend.asarray([1], dtype="torch.int").dtype == backend.int32
assert backend.asarray([1], dtype="torch.long").dtype == backend.int64
assert backend.asarray([1.0 + 0.0j], dtype="torch.cfloat").dtype == backend.complex64
assert backend.asarray([1.0], dtype="float").dtype == backend.float64
assert backend.asarray([1], dtype="int").dtype == backend.int64
print("ok")
"""
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_raw_pytorch_array_resolves_torch_dtype_alias_strings_after_import():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = """
import pyrecest
import pyrecest._backend.pytorch as raw_pytorch

assert raw_pytorch.array([1.0], dtype="torch.float").dtype == raw_pytorch.float32
assert raw_pytorch.array([1.0], dtype="torch.double").dtype == raw_pytorch.float64
assert raw_pytorch.array([1], dtype="torch.int").dtype == raw_pytorch.int32
assert raw_pytorch.array([1], dtype="torch.long").dtype == raw_pytorch.int64
assert raw_pytorch.array([1.0 + 0.0j], dtype="torch.cfloat").dtype == raw_pytorch.complex64
assert raw_pytorch.array([1.0], dtype="float").dtype == raw_pytorch.float64
assert raw_pytorch.array([1], dtype="int").dtype == raw_pytorch.int64
print("ok")
"""
    result = run_backend_code("numpy", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
