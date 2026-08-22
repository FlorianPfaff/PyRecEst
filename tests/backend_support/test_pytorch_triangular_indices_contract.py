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
def test_pytorch_triangular_indices_follow_numpy_contract():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import numpy as np
import pyrecest.backend as backend
import pyrecest._backend.pytorch as pytorch_backend

for module in (backend, pytorch_backend):
    for helper_name in ("tril_indices", "triu_indices"):
        helper = getattr(module, helper_name)
        numpy_helper = getattr(np, helper_name)

        for args, kwargs in (
            ((3,), {}),
            ((3,), {"k": 1}),
            ((2,), {"k": -1, "m": 4}),
            ((-1,), {}),
            ((3,), {"m": -1}),
        ):
            result = helper(*args, **kwargs)
            expected = numpy_helper(*args, **kwargs)
            assert isinstance(result, tuple)
            assert len(result) == 2
            for actual_axis, expected_axis in zip(result, expected):
                assert module.to_numpy(actual_axis).tolist() == expected_axis.tolist()
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=_backend_test_env("pytorch"),
    )


@pytest.mark.backend_portable
def test_raw_pytorch_triangular_indices_are_patched_with_numpy_public_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import numpy as np
import pyrecest.backend as backend
import pyrecest._backend.pytorch as pytorch_backend
import torch

assert getattr(backend, "__backend_name__", None) == "numpy"

rows, cols = pytorch_backend.tril_indices(3)
expected_rows, expected_cols = np.tril_indices(3)
assert pytorch_backend.to_numpy(rows).tolist() == expected_rows.tolist()
assert pytorch_backend.to_numpy(cols).tolist() == expected_cols.tolist()

rows, cols = pytorch_backend.triu_indices(2, k=1, m=4, dtype=torch.int32)
expected_rows, expected_cols = np.triu_indices(2, k=1, m=4)
assert rows.dtype == torch.int32
assert cols.dtype == torch.int32
assert pytorch_backend.to_numpy(rows).tolist() == expected_rows.tolist()
assert pytorch_backend.to_numpy(cols).tolist() == expected_cols.tolist()
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=_backend_test_env("numpy"),
    )
