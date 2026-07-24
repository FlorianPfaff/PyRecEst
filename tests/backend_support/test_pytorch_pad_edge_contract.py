import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable

EXPECTED_EDGE_PADDED = [
    [1, 1, 2, 2],
    [1, 1, 2, 2],
    [3, 3, 4, 4],
    [3, 3, 4, 4],
    [3, 3, 4, 4],
]


def test_raw_pytorch_pad_accepts_numpy_style_edge_mode_after_import():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = f"""
import pyrecest  # noqa: F401  # triggers raw-backend compatibility patches
import torch
import pyrecest._backend.pytorch as raw_pytorch_backend

values = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
result = raw_pytorch_backend.pad(values, ((1, 2), (1, 1)), mode="edge")
assert result.tolist() == {EXPECTED_EDGE_PADDED!r}

one_dimensional = raw_pytorch_backend.pad(
    torch.tensor([1, 2], dtype=torch.int64),
    (1, 2),
    mode="edge",
)
assert one_dimensional.tolist() == [1, 1, 2, 2, 2]
print("ok")
"""
    result = run_backend_code("numpy", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_public_pytorch_pad_accepts_numpy_style_edge_mode():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = f"""
import pyrecest.backend as backend

result = backend.pad(
    backend.array([[1, 2], [3, 4]], dtype=backend.int64),
    ((1, 2), (1, 1)),
    mode="edge",
)
assert backend.to_numpy(result).tolist() == {EXPECTED_EDGE_PADDED!r}
print("ok")
"""
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
