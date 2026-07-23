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
def test_common_pytorch_min_accepts_integer_scalar_axis_tuples():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = r"""
import numpy as np
import torch

import pyrecest  # noqa: F401
from pyrecest._backend import _common

values = torch.tensor([[3.0, 1.0], [2.0, 4.0]])
for axis in ((np.asarray(0),), (torch.tensor(0),)):
    result = _common.min(values, axis=axis)
    assert result.tolist() == [2.0, 1.0]

for axis in ((np.asarray(True),), (torch.tensor(True),)):
    try:
        _common.min(values, axis=axis)
    except TypeError:
        continue
    raise AssertionError(f"accepted boolean axis tuple {axis!r}")
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=_backend_subprocess_env("numpy"),
    )
