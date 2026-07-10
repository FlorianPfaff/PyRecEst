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


_REDUCTION_SCALAR_AXIS_CODE = r'''
import numpy as np
import torch


SCALAR_AXES = (
    np.asarray(0),
    np.asarray(-2),
    torch.tensor(0),
    torch.tensor(-2),
)


def assert_integer_scalar_axes_supported(module):
    values = module.array([[1.0, 2.0], [3.0, 4.0]])
    expected = {
        "sum": [4.0, 6.0],
        "prod": [3.0, 8.0],
        "mean": [2.0, 3.0],
        "std": [1.0, 1.0],
    }

    for axis in SCALAR_AXES:
        for helper_name, expected_values in expected.items():
            result = getattr(module, helper_name)(values, axis=axis)
            assert module.to_numpy(result).tolist() == expected_values

    for dim in SCALAR_AXES:
        for helper_name in ("sum", "mean", "std"):
            result = getattr(module, helper_name)(values, dim=dim)
            assert module.to_numpy(result).tolist() == expected[helper_name]
'''


@pytest.mark.backend_portable
def test_public_pytorch_reductions_accept_integer_scalar_axes():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = (
        _REDUCTION_SCALAR_AXIS_CODE
        + """
import pyrecest.backend as backend

assert backend.__backend_name__ == "pytorch"
assert_integer_scalar_axes_supported(backend)
"""
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=_backend_subprocess_env("pytorch"),
    )


@pytest.mark.backend_portable
def test_raw_pytorch_reductions_are_patched_under_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = (
        _REDUCTION_SCALAR_AXIS_CODE
        + """
import pyrecest  # noqa: F401
import pyrecest.backend as public_backend
import pyrecest._backend.pytorch as raw_pytorch

assert public_backend.__backend_name__ == "numpy"
assert_integer_scalar_axes_supported(raw_pytorch)
"""
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=_backend_subprocess_env("numpy"),
    )
