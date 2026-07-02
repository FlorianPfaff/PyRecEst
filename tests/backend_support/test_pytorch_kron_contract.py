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
@pytest.mark.parametrize("backend_name", ["numpy", "pytorch"])
def test_pytorch_kron_accepts_array_like_inputs_under_public_backend(backend_name):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    env = _backend_subprocess_env(backend_name)

    code = f"""
import numpy as np
import numpy.testing as npt
import pyrecest.backend as public_backend
import pyrecest._backend.pytorch as raw_pytorch

assert public_backend.__backend_name__ == {backend_name!r}

left = [[1, 2], [3, 4]]
right = [0, 5]
expected = np.kron(np.asarray(left), np.asarray(right))

raw_result = raw_pytorch.kron(left, right)
npt.assert_array_equal(raw_pytorch.to_numpy(raw_result), expected)

mixed_result = raw_pytorch.kron(left, raw_pytorch.array(right))
npt.assert_array_equal(raw_pytorch.to_numpy(mixed_result), expected)

raw_out = raw_pytorch.empty_like(raw_result)
raw_returned = raw_pytorch.kron(left, right, out=raw_out)
assert raw_returned is raw_out
npt.assert_array_equal(raw_pytorch.to_numpy(raw_out), expected)

if public_backend.__backend_name__ == "pytorch":
    public_result = public_backend.kron(left, right)
    npt.assert_array_equal(public_backend.to_numpy(public_result), expected)

    out = public_backend.empty_like(public_result)
    returned = public_backend.kron(left, right, out=out)
    assert returned is out
    npt.assert_array_equal(public_backend.to_numpy(out), expected)
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
