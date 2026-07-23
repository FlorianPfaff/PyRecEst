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
def test_pytorch_fft_aliases_accept_matching_numpy_array_axes():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import numpy as np
import numpy.testing as npt
import pyrecest.backend as backend
from pyrecest._backend import pytorch as pytorch_backend

values_np = np.arange(4).reshape(2, 2)
values = backend.array(values_np)

for fft_namespace, as_numpy in (
    (backend.fft, backend.to_numpy),
    (pytorch_backend.fft, pytorch_backend.to_numpy),
):
    result = fft_namespace.fftn(
        values,
        axes=np.array([0, 1]),
        dim=np.array([0, 1]),
    )
    npt.assert_allclose(as_numpy(result), np.fft.fftn(values_np, axes=(0, 1)))

    shifted = fft_namespace.fftshift(
        values,
        axes=np.array([0, 1]),
        dim=np.array([0, 1]),
    )
    npt.assert_array_equal(as_numpy(shifted), np.fft.fftshift(values_np, axes=(0, 1)))

    try:
        fft_namespace.fftn(values, axes=np.array([0]), dim=np.array([1]))
    except TypeError:
        pass
    else:
        raise AssertionError("conflicting FFT axis aliases were accepted")
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, env=_backend_test_env("pytorch")
    )
