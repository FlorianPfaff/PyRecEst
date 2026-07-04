import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def test_public_jax_one_hot_accepts_shared_keyword_contract():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed")

    code = """
import pyrecest.backend as backend

result = backend.one_hot(labels=[1, 0], num_classes=3)
assert result.shape == (2, 3)
assert str(backend.to_numpy(result).dtype) == "uint8"
assert backend.to_numpy(result).tolist() == [[0, 1, 0], [1, 0, 0]]
print("ok")
"""
    result = run_backend_code("jax", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_raw_jax_one_hot_accepts_shared_keyword_contract_after_import():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed")

    code = """
import pyrecest  # noqa: F401  # triggers raw-backend compatibility patches
import pyrecest._backend.jax as raw_jax

result = raw_jax.one_hot(labels=[2, 1], num_classes=4)
assert result.shape == (2, 4)
assert raw_jax.to_numpy(result).tolist() == [[0, 0, 1, 0], [0, 1, 0, 0]]
print("ok")
"""
    result = run_backend_code("numpy", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
