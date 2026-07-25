"""Regression tests for the PyTorch backend sort facade fallback."""

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

_CHECK = """
import pyrecest.backend as backend

values = backend.sort([3, 1, 2])
assert backend.to_numpy(values).tolist() == [1, 2, 3]

flat = backend.sort([[3, 1], [2, 4]], axis=None)
assert backend.to_numpy(flat).tolist() == [1, 2, 3, 4]

descending = backend.sort([1, 3, 2], descending=True)
assert backend.to_numpy(descending).tolist() == [3, 2, 1]

stable_values = backend.sort([2, 1, 2], kind="stable")
assert backend.to_numpy(stable_values).tolist() == [1, 2, 2]
print("ok")
"""


@pytest.mark.backend_portable
def test_pytorch_backend_imports_and_sort_matches_numpy_style_contract():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("pytorch", _CHECK)

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
