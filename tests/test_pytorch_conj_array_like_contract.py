import pytest
from tests.support.backend_runner import run_backend_code


def test_public_pytorch_conj_accepts_array_like_inputs():
    pytest.importorskip("torch")

    code = """
import pyrecest.backend as backend

values = backend.conj([1.0 + 2.0j, 3.0 - 4.0j])
converted = backend.to_numpy(values)
assert converted.tolist() == [1.0 - 2.0j, 3.0 + 4.0j]
"""
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr


def test_raw_pytorch_conj_accepts_array_like_under_default_backend():
    pytest.importorskip("torch")

    code = """
import pyrecest
import pyrecest._backend.pytorch as raw_pytorch

result = raw_pytorch.conj([1.0 + 2.0j, 3.0 - 4.0j])
assert raw_pytorch.to_numpy(result).tolist() == [1.0 - 2.0j, 3.0 + 4.0j]
"""
    result = run_backend_code("numpy", code)
    assert result.returncode == 0, result.stderr
