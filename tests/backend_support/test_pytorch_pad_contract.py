import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable

EXPECTED_PADDED = [
    [7, 5, 5, 8],
    [7, 1, 2, 8],
    [7, 3, 4, 8],
    [7, 6, 6, 8],
]

BAD_PAD_WIDTHS = [
    "1.5",
    "((0, 1.25),)",
    "True",
    "[True, False]",
    "np.array([1, 2], dtype=np.uint8)",
]


def test_raw_pytorch_pad_accepts_numpy_style_constant_values_after_import():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = f"""
import pyrecest  # noqa: F401  # triggers raw-backend compatibility patches
import torch
import pyrecest._backend.pytorch as raw_pytorch_backend

values = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
result = raw_pytorch_backend.pad(
    values,
    ((1, 1), (1, 1)),
    mode="constant",
    constant_values=((5, 6), (7, 8)),
)
assert result.tolist() == {EXPECTED_PADDED!r}

one_dimensional = raw_pytorch_backend.pad(
    torch.tensor([1, 2], dtype=torch.int64),
    (1, 2),
    mode="constant",
    constant_values=(9, 10),
)
assert one_dimensional.tolist() == [9, 1, 2, 10, 10]
print("ok")
"""
    result = run_backend_code("numpy", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_public_pytorch_pad_accepts_numpy_style_constant_values():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = f"""
import pyrecest.backend as backend

result = backend.pad(
    backend.array([[1, 2], [3, 4]], dtype=backend.int64),
    ((1, 1), (1, 1)),
    mode="constant",
    constant_values=((5, 6), (7, 8)),
)
assert backend.to_numpy(result).tolist() == {EXPECTED_PADDED!r}
print("ok")
"""
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_raw_pytorch_pad_rejects_non_integral_pad_width_after_import():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = f"""
import numpy as np
import pyrecest  # noqa: F401  # triggers raw-backend compatibility patches
import pyrecest._backend.pytorch as raw_pytorch_backend

bad_pad_widths = [{", ".join(BAD_PAD_WIDTHS)}]
for bad_pad_width in bad_pad_widths:
    try:
        raw_pytorch_backend.pad([1, 2], bad_pad_width, mode="constant")
    except TypeError as exc:
        assert "pad_width" in str(exc)
        assert "integral" in str(exc)
    else:
        raise AssertionError(f"accepted invalid pad_width: {{bad_pad_width!r}}")
print("ok")
"""
    result = run_backend_code("numpy", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_public_pytorch_pad_rejects_non_integral_pad_width():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = f"""
import numpy as np
import pyrecest.backend as backend

values = backend.array([1, 2], dtype=backend.int64)
bad_pad_widths = [{", ".join(BAD_PAD_WIDTHS)}]
for bad_pad_width in bad_pad_widths:
    try:
        backend.pad(values, bad_pad_width, mode="constant")
    except TypeError as exc:
        assert "pad_width" in str(exc)
        assert "integral" in str(exc)
    else:
        raise AssertionError(f"accepted invalid pad_width: {{bad_pad_width!r}}")
print("ok")
"""
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_raw_pytorch_concatenate_axis_none_flattens_after_import():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = """
import pyrecest  # noqa: F401  # triggers raw-backend compatibility patches
import pyrecest._backend.pytorch as raw_pytorch_backend

result = raw_pytorch_backend.concatenate(
    (raw_pytorch_backend.array([[1, 2]]), raw_pytorch_backend.array([[3], [4]])),
    axis=None,
)
assert raw_pytorch_backend.to_numpy(result).tolist() == [1, 2, 3, 4]
print("ok")
"""
    result = run_backend_code("numpy", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_public_pytorch_concatenate_axis_none_flattens_inputs():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = """
import pyrecest.backend as backend

result = backend.concatenate(
    (backend.array([[1, 2]]), backend.array([[3], [4]])),
    axis=None,
)
assert backend.to_numpy(result).tolist() == [1, 2, 3, 4]
print("ok")
"""
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
