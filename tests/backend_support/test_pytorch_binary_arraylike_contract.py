import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def _require_torch():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")


def test_public_pytorch_mod_and_power_accept_arraylike_second_operands():
    _require_torch()

    code = """
import pyrecest.backend as backend

assert getattr(backend, "__backend_name__", None) == "pytorch"

mod_result = backend.mod([5, 7, 9], [2, 3, 4])
power_result = backend.power([2, 3, 4], [3, 2, 1])
scalar_result = backend.mod([5, 7, 9], 2)

assert backend.to_numpy(mod_result).tolist() == [1, 1, 1]
assert backend.to_numpy(power_result).tolist() == [8, 9, 4]
assert backend.to_numpy(scalar_result).tolist() == [1, 1, 1]
print("ok")
"""
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_raw_pytorch_mod_and_power_accept_arraylike_second_operands_after_import():
    _require_torch()

    code = """
import pyrecest  # noqa: F401  # triggers backend compatibility patches
import pyrecest._backend.pytorch as raw_pytorch

mod_result = raw_pytorch.mod([5, 7, 9], [2, 3, 4])
power_result = raw_pytorch.power([2, 3, 4], [3, 2, 1])
scalar_result = raw_pytorch.power([2, 3, 4], 2)

assert raw_pytorch.to_numpy(mod_result).tolist() == [1, 1, 1]
assert raw_pytorch.to_numpy(power_result).tolist() == [8, 9, 4]
assert raw_pytorch.to_numpy(scalar_result).tolist() == [4, 9, 16]
print("ok")
"""
    result = run_backend_code("numpy", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
