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


_REDUCTION_BOOL_AXIS_CODE = r'''
import numpy as np
import torch

axis_values = [
    True,
    False,
    np.bool_(True),
    np.asarray(True),
    np.asarray([True]),
    (True,),
    [False],
    torch.tensor(True),
    torch.tensor([True]),
    (torch.tensor(False),),
]


def assert_bool_axes_rejected(module):
    values = module.array([[0, 1], [2, 3]])
    for helper_name in ("any", "all", "count_nonzero", "max", "mean", "std"):
        helper = getattr(module, helper_name)
        for axis in axis_values:
            try:
                helper(values, axis=axis)
            except TypeError:
                continue
            raise AssertionError(f"{helper_name} accepted boolean axis {axis!r}")

    for helper_name in ("mean", "std"):
        helper = getattr(module, helper_name)
        for dim in axis_values:
            try:
                helper(values, dim=dim)
            except TypeError:
                continue
            raise AssertionError(f"{helper_name} accepted boolean dim {dim!r}")

    assert module.to_numpy(module.any(values, axis=1)).tolist() == [True, True]
    assert module.to_numpy(module.max(values, axis=0)).tolist() == [2, 3]
    assert module.to_numpy(module.mean(values, axis=0)).tolist() == [1.0, 2.0]
'''


@pytest.mark.backend_portable
def test_public_pytorch_reduction_helpers_reject_boolean_axes():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = (
        _REDUCTION_BOOL_AXIS_CODE
        + """
import pyrecest.backend as backend

assert backend.__backend_name__ == "pytorch"
assert_bool_axes_rejected(backend)
"""
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=_backend_subprocess_env("pytorch"),
    )


@pytest.mark.backend_portable
def test_raw_pytorch_reduction_helpers_are_patched_under_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = (
        _REDUCTION_BOOL_AXIS_CODE
        + """
import pyrecest  # noqa: F401
import pyrecest.backend as public_backend
import pyrecest._backend.pytorch as raw_pytorch

assert public_backend.__backend_name__ == "numpy"
assert_bool_axes_rejected(raw_pytorch)
"""
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=_backend_subprocess_env("numpy"),
    )
