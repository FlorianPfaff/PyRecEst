import importlib.util
import os
import subprocess
import sys

import pyrecest.backend as backend
import pytest
from pyrecest.backend_tools import get_backend_name


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_cumulative_out_contract_default_backend():
    if get_backend_name() == "jax":
        pytest.skip("JAX arrays do not support NumPy-style out mutation")

    values = backend.reshape(backend.arange(1, 7), (2, 3))
    out_sum = backend.zeros((2, 3), dtype=values.dtype)
    out_prod = backend.zeros((2, 3), dtype=values.dtype)

    result_sum = backend.cumsum(values, axis=1, out=out_sum)
    result_prod = backend.cumprod(values, axis=1, out=out_prod)

    assert result_sum is out_sum
    assert result_prod is out_prod
    assert _to_python(result_sum) == [[1, 3, 6], [4, 9, 15]]
    assert _to_python(result_prod) == [[1, 2, 6], [4, 20, 120]]


@pytest.mark.backend_portable
def test_pytorch_cumulative_out_contract():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "pytorch"
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    code = """
import pyrecest.backend as backend
values = backend.reshape(backend.arange(1, 7), (2, 3))
out_sum = backend.zeros((2, 3), dtype=values.dtype)
out_prod = backend.zeros((2, 3), dtype=values.dtype)
result_sum = backend.cumsum(values, axis=1, out=out_sum)
result_prod = backend.cumprod(values, axis=1, out=out_prod)
assert result_sum is out_sum
assert result_prod is out_prod
assert backend.to_numpy(result_sum).tolist() == [[1, 3, 6], [4, 9, 15]]
assert backend.to_numpy(result_prod).tolist() == [[1, 2, 6], [4, 20, 120]]
flat = backend.cumsum(values, out=backend.zeros((6,), dtype=values.dtype))
assert backend.to_numpy(flat).tolist() == [1, 3, 6, 10, 15, 21]
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
