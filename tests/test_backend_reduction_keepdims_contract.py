import importlib.util

import numpy as np
import pyrecest.backend as backend
import pytest
from tests.support.backend_runner import run_backend_code


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_reductions_accept_keepdims_keyword():
    values = backend.array([[[0, 1], [2, 0]], [[3, 4], [0, 0]]])

    assert _to_python(backend.any(values, axis=(0, 2), keepdims=True)) == [
        [[True], [True]]
    ]
    assert _to_python(backend.all(values > -1, axis=(0, 2), keepdims=True)) == [
        [[True], [True]]
    ]
    assert _to_python(backend.max(values, axis=(0, 2), keepdims=True)) == [[[4], [2]]]
    assert _to_python(backend.amin(values, axis=(0, 2), keepdims=True)) == [[[0], [0]]]
    assert _to_python(backend.min(values, axis=(0, 2), keepdims=True)) == [[[0], [0]]]
    assert _to_python(backend.prod(values + 1, axis=(0, 2), keepdims=True)) == [
        [[40], [3]]
    ]


def test_reductions_accept_numpy_scalar_array_axis():
    values = backend.array([[1, 2, 3], [4, 5, 6]])
    axis = np.array(1)

    assert _to_python(backend.sum(values, axis=axis)) == [6, 15]
    assert _to_python(backend.mean(values, axis=axis)) == [2.0, 5.0]
    assert _to_python(backend.max(values, axis=axis)) == [3, 6]
    assert _to_python(backend.prod(values, axis=axis)) == [6, 120]
    assert _to_python(backend.quantile(values, 0.5, axis=axis)) == [2.0, 5.0]


@pytest.mark.backend_portable
def test_pytorch_reductions_accept_keepdims_keyword():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend

values = backend.array([[[0, 1], [2, 0]], [[3, 4], [0, 0]]])
assert backend.to_numpy(backend.any(values, axis=(0, 2), keepdims=True)).tolist() == [[[True], [True]]]
assert backend.to_numpy(backend.all(values > -1, axis=(0, 2), keepdims=True)).tolist() == [[[True], [True]]]
assert backend.to_numpy(backend.max(values, axis=(0, 2), keepdims=True)).tolist() == [[[4], [2]]]
assert backend.to_numpy(backend.amin(values, axis=(0, 2), keepdims=True)).tolist() == [[[0], [0]]]
assert backend.to_numpy(backend.min(values, axis=(0, 2), keepdims=True)).tolist() == [[[0], [0]]]
assert backend.to_numpy(backend.prod(values + 1, axis=(0, 2), keepdims=True)).tolist() == [[[40], [3]]]
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.backend_portable
def test_pytorch_reductions_accept_numpy_scalar_array_axis():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import numpy as np
import pyrecest.backend as backend

values = backend.array([[1, 2, 3], [4, 5, 6]])
axis = np.array(1)
assert backend.to_numpy(backend.sum(values, axis=axis)).tolist() == [6, 15]
assert backend.to_numpy(backend.mean(values, axis=axis)).tolist() == [2.0, 5.0]
assert backend.to_numpy(backend.max(values, axis=axis)).tolist() == [3, 6]
assert backend.to_numpy(backend.prod(values, axis=axis)).tolist() == [6, 120]
assert backend.to_numpy(backend.quantile(values, 0.5, axis=axis)).tolist() == [2.0, 5.0]
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.backend_portable
def test_raw_pytorch_reductions_accept_numpy_scalar_array_axis_after_import():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import numpy as np
import pyrecest  # noqa: F401 - triggers import-time backend compatibility patches
import pyrecest._backend.pytorch as raw_pytorch

values = raw_pytorch.array([[1, 2, 3], [4, 5, 6]])
axis = np.array(1)
assert raw_pytorch.to_numpy(raw_pytorch.sum(values, axis=axis)).tolist() == [6, 15]
assert raw_pytorch.to_numpy(raw_pytorch.sum(values, dim=axis)).tolist() == [6, 15]
assert raw_pytorch.to_numpy(raw_pytorch.max(values, axis=axis)).tolist() == [3, 6]
assert raw_pytorch.to_numpy(raw_pytorch.quantile(values, 0.5, axis=axis)).tolist() == [2.0, 5.0]
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
