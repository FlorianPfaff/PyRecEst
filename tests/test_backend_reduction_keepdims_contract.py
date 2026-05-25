import importlib.util

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
    assert _to_python(backend.min(values, axis=(0, 2), keepdims=True)) == [[[0], [0]]]
    assert _to_python(backend.prod(values + 1, axis=(0, 2), keepdims=True)) == [
        [[40], [3]]
    ]


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
assert backend.to_numpy(backend.min(values, axis=(0, 2), keepdims=True)).tolist() == [[[0], [0]]]
assert backend.to_numpy(backend.prod(values + 1, axis=(0, 2), keepdims=True)).tolist() == [[[40], [3]]]
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
