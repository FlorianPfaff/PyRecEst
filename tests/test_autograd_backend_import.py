import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_autograd_backend_imports_and_exposes_contract_smoke():
    if importlib.util.find_spec("autograd") is None:
        pytest.skip("autograd is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "autograd"
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    code = """
import pyrecest.backend as backend
assert backend.__backend_name__ == 'autograd'
assert backend.has_autodiff()
assert backend.diag(backend.array([1.0, 2.0])).shape == (2, 2)
assert backend.random.get_state() is not None
assert backend.fft.rfft(backend.array([1.0, 0.0])).shape == (2,)
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
