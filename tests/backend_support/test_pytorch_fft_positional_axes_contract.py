import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_fft_positional_axes_accept_numpy_integer_arrays():
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
import numpy as np
import numpy.testing as npt
import pyrecest.backend as backend

values_np = np.array([[0.0, 1.0], [2.0, 3.0]])
values = backend.array(values_np)

for axis in (np.array(0), np.int64(0)):
    npt.assert_allclose(
        backend.to_numpy(backend.fft.rfft(values, None, axis)),
        np.fft.rfft(values_np, None, axis),
    )
    npt.assert_allclose(
        backend.to_numpy(backend.fft.irfft(values, None, axis)),
        np.fft.irfft(values_np, None, axis),
    )

axes_cases = (
    np.array([0]),
    np.array([0, 1]),
    [np.array(0)],
    (np.array(1),),
)
for axes in axes_cases:
    npt.assert_array_equal(
        backend.to_numpy(backend.fft.fftshift(values, axes)),
        np.fft.fftshift(values_np, axes),
    )
    npt.assert_array_equal(
        backend.to_numpy(backend.fft.ifftshift(values, axes)),
        np.fft.ifftshift(values_np, axes),
    )
    npt.assert_allclose(
        backend.to_numpy(backend.fft.fftn(values, None, axes)),
        np.fft.fftn(values_np, None, axes),
    )
    npt.assert_allclose(
        backend.to_numpy(backend.fft.ifftn(values, None, axes)),
        np.fft.ifftn(values_np, None, axes),
    )

for axes in ((), []):
    npt.assert_array_equal(
        backend.to_numpy(backend.fft.fftshift(values, axes)),
        values_np,
    )
    npt.assert_array_equal(
        backend.to_numpy(backend.fft.ifftshift(values, axes)),
        values_np,
    )
    npt.assert_array_equal(
        backend.to_numpy(backend.fft.fftn(values, None, axes)),
        values_np,
    )
    npt.assert_array_equal(
        backend.to_numpy(backend.fft.ifftn(values, None, axes)),
        values_np,
    )
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
