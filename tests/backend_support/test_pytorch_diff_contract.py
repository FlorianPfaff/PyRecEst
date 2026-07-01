import importlib.util
import os
import subprocess
import sys

import pytest

SCRIPT = """
import pyrecest  # noqa: F401  # triggers raw-backend compatibility patches
import torch
import pyrecest._backend.pytorch as raw_pytorch_backend

values = [[1.0, 3.0, 6.0], [0.0, 5.0, 9.0]]

axis_result = raw_pytorch_backend.diff(values, axis=0)
axis_expected = torch.tensor([[-1.0, 2.0, 3.0]])
assert torch.allclose(axis_result, axis_expected)

prepend_result = raw_pytorch_backend.diff(values, axis=1, prepend=0.0)
prepend_expected = torch.tensor([[1.0, 2.0, 3.0], [0.0, 5.0, 4.0]])
assert torch.allclose(prepend_result, prepend_expected)

append_result = raw_pytorch_backend.diff(values, axis=-1, append=[[10.0], [12.0]])
append_expected = torch.tensor([[2.0, 3.0, 4.0], [5.0, 4.0, 3.0]])
assert torch.allclose(append_result, append_expected)

zero_order = raw_pytorch_backend.diff(5.0, n=0)
assert tuple(zero_order.shape) == ()
assert float(zero_order) == 5.0

try:
    raw_pytorch_backend.diff(5.0)
except ValueError as exc:
    assert "at least one dimensional" in str(exc)
else:  # pragma: no cover - exercised in subprocess
    raise AssertionError("scalar diff with n > 0 should fail")

print("ok")
"""


def test_raw_pytorch_diff_accepts_numpy_style_contract_after_import():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    env = os.environ.copy()
    env.pop("PYRECEST_BACKEND", None)
    completed = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        capture_output=True,
        env=env,
        text=True,
        timeout=30.0,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout


def test_public_pytorch_diff_accepts_numpy_style_contract():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "pytorch"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import pyrecest.backend as backend

result = backend.diff([[1.0, 3.0, 6.0], [0.0, 5.0, 9.0]], axis=1, prepend=0.0)
expected = backend.array([[1.0, 2.0, 3.0], [0.0, 5.0, 4.0]])
assert backend.allclose(result, expected)
print("ok")
""",
        ],
        capture_output=True,
        env=env,
        text=True,
        timeout=30.0,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout
