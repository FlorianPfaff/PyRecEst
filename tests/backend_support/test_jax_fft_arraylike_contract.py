import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_jax_fft_helpers_accept_python_lists():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "jax"
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    code = """
import pyrecest.backend as backend

values = [1.0, 2.0, 3.0, 4.0]
spectrum = backend.fft.rfft(values)
expected = backend.fft.rfft(backend.array(values))
assert tuple(backend.shape(spectrum)) == (3,)
assert bool(backend.to_numpy(backend.allclose(spectrum, expected)))

shifted = backend.fft.fftshift(values)
assert backend.to_numpy(shifted).tolist() == [3.0, 4.0, 1.0, 2.0]
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
