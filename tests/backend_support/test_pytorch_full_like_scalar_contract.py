import importlib.util
import os
import subprocess
import sys

import pytest


def _subprocess_env(selected_backend):
    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = selected_backend
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )
    return env


@pytest.mark.backend_portable
@pytest.mark.parametrize(
    ("selected_backend", "helper_expression"),
    [
        ("pytorch", "backend"),
        ("numpy", "raw_pytorch"),
    ],
)
def test_pytorch_full_like_accepts_numpy_scalar_fill_values(
    selected_backend,
    helper_expression,
):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = f"""
import numpy as np
import pyrecest  # noqa: F401  # triggers raw-backend compatibility patches
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch

helper = {helper_expression}
source = [[1, 2], [3, 4]]
fill_value = np.asarray(7)
assert fill_value.shape == ()

filled = helper.full_like(source, fill_value)
assert helper.to_numpy(filled).tolist() == [[7, 7], [7, 7]]

print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        env=_subprocess_env(selected_backend),
        text=True,
        timeout=30.0,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout
