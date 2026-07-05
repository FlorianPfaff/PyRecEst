import importlib.util
import os
import subprocess
import sys

import pytest


def _backend_test_env(backend_name):
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
def test_pytorch_take_rejects_boolean_and_float_axes():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest.backend as backend

values = backend.asarray([[1, 2], [3, 4]])

for axis in (True, False, 1.0):
    try:
        backend.take(values, [0], axis=axis)
    except TypeError:
        pass
    else:
        raise AssertionError(f"backend.take accepted invalid axis {axis!r}")

result = backend.take(values, [0], axis=backend.asarray(1))
assert backend.to_numpy(result).tolist() == [[1], [3]]
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=_backend_test_env("pytorch"),
    )


@pytest.mark.backend_portable
def test_raw_pytorch_take_rejects_boolean_and_float_axes_with_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch

assert getattr(backend, "__backend_name__", None) == "numpy"

values = raw_pytorch.asarray([[1, 2], [3, 4]])

for axis in (True, False, 1.0):
    try:
        raw_pytorch.take(values, [0], axis=axis)
    except TypeError:
        pass
    else:
        raise AssertionError(f"raw_pytorch.take accepted invalid axis {axis!r}")

result = raw_pytorch.take(values, [0], axis=raw_pytorch.asarray(1))
assert raw_pytorch.to_numpy(result).tolist() == [[1], [3]]
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=_backend_test_env("numpy"),
    )
