import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_raw_pytorch_broadcast_arrays_accepts_array_like_with_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "numpy"
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    code = """
import pyrecest  # noqa: F401
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_backend

assert backend.__backend_name__ == "numpy"

left, right = raw_backend.broadcast_arrays([1, 2], raw_backend.array([[3], [4]]))
assert raw_backend.to_numpy(left).tolist() == [[1, 2], [1, 2]]
assert raw_backend.to_numpy(right).tolist() == [[3, 3], [4, 4]]

scalar, matrix = raw_backend.broadcast_arrays(5, [[1, 2], [3, 4]])
assert raw_backend.to_numpy(scalar).tolist() == [[5, 5], [5, 5]]
assert raw_backend.to_numpy(matrix).tolist() == [[1, 2], [3, 4]]
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
