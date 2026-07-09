import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest
from pyrecest.backend_support._pytorch_sort_numpy_contract import (
    resolve_sort_stability,
)


@pytest.mark.parametrize(
    ("kind", "stable"),
    [
        ("stable", True),
        ("stable", np.bool_(True)),
        ("mergesort", False),
        ("quicksort", False),
        ("heapsort", np.bool_(False)),
    ],
)
def test_sort_kind_and_stable_are_mutually_exclusive_like_numpy(kind, stable):
    with pytest.raises(ValueError, match="kind.*stable"):
        resolve_sort_stability(kind, stable)


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
def test_public_pytorch_sort_axis_none_flattens_like_numpy():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    env = _backend_subprocess_env("pytorch")

    code = """
import pyrecest.backend as backend

assert backend.__backend_name__ == "pytorch"

flat_result = backend.sort([[3, 1], [2, 0]], axis=None)
assert backend.to_numpy(flat_result).tolist() == [0, 1, 2, 3]

axis_result = backend.sort([[3, 1], [2, 0]], axis=0)
assert backend.to_numpy(axis_result).tolist() == [[2, 0], [3, 1]]

descending_result = backend.sort([[3, 1], [2, 0]], axis=None, descending=True)
assert backend.to_numpy(descending_result).tolist() == [3, 2, 1, 0]

stable_result = backend.sort([[3, 1], [2, 0]], axis=None, kind="stable")
assert backend.to_numpy(stable_result).tolist() == [0, 1, 2, 3]
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.mark.backend_portable
def test_raw_pytorch_sort_axis_none_is_patched_under_default_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    env = _backend_subprocess_env("numpy")

    code = """
import pyrecest  # noqa: F401
import pyrecest.backend as public_backend
import pyrecest._backend.pytorch as raw_pytorch

assert public_backend.__backend_name__ == "numpy"

flat_result = raw_pytorch.sort([[3, 1], [2, 0]], axis=None)
assert raw_pytorch.to_numpy(flat_result).tolist() == [0, 1, 2, 3]

axis_result = raw_pytorch.sort([[3, 1], [2, 0]], axis=0)
assert raw_pytorch.to_numpy(axis_result).tolist() == [[2, 0], [3, 1]]

descending_result = raw_pytorch.sort([[3, 1], [2, 0]], axis=None, descending=True)
assert raw_pytorch.to_numpy(descending_result).tolist() == [3, 2, 1, 0]

stable_result = raw_pytorch.sort([[3, 1], [2, 0]], axis=None, kind="stable")
assert raw_pytorch.to_numpy(stable_result).tolist() == [0, 1, 2, 3]
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_sort_stability_rejects_kind_and_stable_combinations():
    stable_values = (True, False, np.bool_(True), np.bool_(False), 1, 0)

    for kind in ("stable", "mergesort", "quicksort", "heapsort"):
        for stable in stable_values:
            with pytest.raises(ValueError, match="kind.*stable"):
                resolve_sort_stability(kind, stable)
