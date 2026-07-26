import importlib.util
import os
import subprocess
import sys

import pytest

SCRIPT = """
import numpy as np
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch_backend
import torch

expected = torch.tensor([0, 2, 4])
raw_result = raw_pytorch_backend.arange(np.asarray(5), step=np.asarray(2))
public_result = backend.arange(5, step=2)

assert torch.equal(raw_result, expected)
assert torch.equal(public_result, expected)
print("ok")
"""


def test_pytorch_arange_honors_step_with_single_positional_argument():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "pytorch"
    completed = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        capture_output=True,
        env=env,
        text=True,
        timeout=30.0,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout
