import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_maximum_minimum_accept_out_keyword():
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
from pyrecest._backend import pytorch as pytorch_backend

left = backend.array([1.0, 4.0])
right = backend.array([2.0, 3.0])

for func, expected in (
    (backend.maximum, [2.0, 4.0]),
    (pytorch_backend.maximum, [2.0, 4.0]),
    (backend.minimum, [1.0, 3.0]),
    (pytorch_backend.minimum, [1.0, 3.0]),
):
    out = backend.empty_like(left)
    returned = func(left, right, out=out)
    assert returned is out
    assert backend.to_numpy(out).tolist() == expected
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
