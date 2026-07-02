import importlib.util
import os
import subprocess
import sys

import pytest


def _backend_subprocess_env(backend_name):
    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = backend_name
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )
    return env


@pytest.mark.backend_portable
def test_jax_matmul_honors_out_shape_contract():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax is not installed")

    env = _backend_subprocess_env("jax")

    code = """
import pyrecest.backend as backend
import pyrecest._backend.jax as raw_jax_backend

left = [[1.0, 2.0], [3.0, 4.0]]
right = [[1.0, 0.0], [0.0, 1.0]]

out = backend.zeros((2, 2))
returned = backend.matmul(left, right, out=out)
assert backend.to_numpy(returned).tolist() == [[1.0, 2.0], [3.0, 4.0]]

bad_out = backend.zeros((1, 1))
try:
    backend.matmul(left, right, out=bad_out)
except (TypeError, ValueError):
    pass
else:
    raise AssertionError("JAX backend.matmul ignored incompatible out shape")

raw_bad_out = raw_jax_backend.zeros((1, 1))
try:
    raw_jax_backend.matmul(left, right, out=raw_bad_out)
except (TypeError, ValueError):
    pass
else:
    raise AssertionError("raw JAX matmul ignored incompatible out shape")
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.mark.backend_portable
def test_raw_jax_matmul_honors_out_when_public_backend_is_numpy():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax is not installed")

    env = _backend_subprocess_env("numpy")

    code = """
import pyrecest.backend as public_backend
import pyrecest._backend.jax as raw_jax_backend

assert public_backend.__backend_name__ == "numpy"

left = [[1.0, 2.0], [3.0, 4.0]]
right = [[1.0, 0.0], [0.0, 1.0]]

out = raw_jax_backend.zeros((2, 2))
returned = raw_jax_backend.matmul(left, right, out=out)
assert raw_jax_backend.to_numpy(returned).tolist() == [[1.0, 2.0], [3.0, 4.0]]

bad_out = raw_jax_backend.zeros((1, 1))
try:
    raw_jax_backend.matmul(left, right, out=bad_out)
except (TypeError, ValueError):
    pass
else:
    raise AssertionError("raw JAX matmul ignored incompatible out shape under NumPy backend")
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
