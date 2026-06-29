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
def test_low_level_pytorch_tile_uses_numpy_repetitions_with_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    env = _backend_subprocess_env("numpy")

    code = """
import pyrecest.backend as public_backend
from pyrecest._backend import pytorch as pytorch_backend

assert public_backend.__backend_name__ == "numpy"
values = pytorch_backend.array([[1, 2], [3, 4]])

scalar_result = pytorch_backend.tile(values, 2)
assert tuple(scalar_result.shape) == (2, 4)
assert pytorch_backend.to_numpy(scalar_result).tolist() == [[1, 2, 1, 2], [3, 4, 3, 4]]

array_result = pytorch_backend.tile(values, pytorch_backend.array([2, 1]))
assert tuple(array_result.shape) == (4, 2)
assert pytorch_backend.to_numpy(array_result).tolist() == [[1, 2], [3, 4], [1, 2], [3, 4]]

empty_result = pytorch_backend.tile(values, ())
assert tuple(empty_result.shape) == (2, 2)
assert pytorch_backend.to_numpy(empty_result).tolist() == [[1, 2], [3, 4]]
assert empty_result is not values

for bad_reps in (1.5, [2.5, 1], "2", pytorch_backend.array([2.5, 1.0])):
    try:
        pytorch_backend.tile(values, bad_reps)
    except TypeError:
        pass
    else:
        raise AssertionError(f"tile accepted non-integer repetitions {bad_reps!r}")
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
