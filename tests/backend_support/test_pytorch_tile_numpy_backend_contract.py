import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_tile_module_uses_torch_tensor_under_numpy_backend():
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
import pyrecest.backend as public_backend
from pyrecest._backend import pytorch as pytorch_backend

assert public_backend.__backend_name__ == "numpy"
values = pytorch_backend.array([[1, 2], [3, 4]])

result = pytorch_backend.tile(values, 2)
assert tuple(result.shape) == (2, 4)
assert pytorch_backend.to_numpy(result).tolist() == [[1, 2, 1, 2], [3, 4, 3, 4]]
assert result.device == values.device
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
