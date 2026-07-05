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
def test_pytorch_triangular_helpers_accept_array_like_inputs():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest.backend as backend
import pyrecest._backend.pytorch as pytorch_backend

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

for module in (backend, pytorch_backend):
    diag = module.vec_to_diag([1, 2, 3])
    assert module.to_numpy(diag).tolist() == [[1, 0, 0], [0, 2, 0], [0, 0, 3]]

    lower = module.tril_to_vec(matrix)
    assert module.to_numpy(lower).tolist() == [1, 4, 5, 7, 8, 9]

    upper = module.triu_to_vec(matrix, k=1)
    assert module.to_numpy(upper).tolist() == [2, 3, 6]
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=_backend_test_env("pytorch"),
    )


@pytest.mark.backend_portable
def test_raw_pytorch_triangular_helpers_with_numpy_public_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest.backend as backend
import pyrecest._backend.pytorch as pytorch_backend

assert getattr(backend, "__backend_name__", None) == "numpy"

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

diag = pytorch_backend.vec_to_diag([1, 2, 3])
assert pytorch_backend.to_numpy(diag).tolist() == [[1, 0, 0], [0, 2, 0], [0, 0, 3]]

lower = pytorch_backend.tril_to_vec(matrix)
assert pytorch_backend.to_numpy(lower).tolist() == [1, 4, 5, 7, 8, 9]

upper = pytorch_backend.triu_to_vec(matrix, k=1)
assert pytorch_backend.to_numpy(upper).tolist() == [2, 3, 6]
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=_backend_test_env("numpy"),
    )
