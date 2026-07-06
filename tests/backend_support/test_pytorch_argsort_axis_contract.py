import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def test_public_pytorch_argsort_rejects_boolean_axes():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = """
import numpy as np
import pyrecest.backend as backend

values = backend.asarray([[3, 1, 2], [6, 4, 5]])
for axis in (True, False, np.bool_(True), np.array(True), backend.asarray(True)):
    try:
        backend.argsort(values, axis=axis)
    except TypeError:
        pass
    else:
        raise AssertionError(f"accepted boolean argsort axis {axis!r}")

try:
    backend.argsort(values, dim=True)
except TypeError:
    pass
else:
    raise AssertionError("accepted boolean argsort dim")

assert backend.to_numpy(backend.argsort(values, axis=1)).tolist() == [[1, 2, 0], [1, 2, 0]]
assert backend.to_numpy(backend.argsort(values, axis=backend.asarray(1))).tolist() == [[1, 2, 0], [1, 2, 0]]
print("ok")
"""
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_raw_pytorch_argsort_rejects_boolean_axes_after_import():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = """
import numpy as np
import pyrecest  # noqa: F401  # triggers raw-backend compatibility patches
import pyrecest._backend.pytorch as raw_pytorch

values = raw_pytorch.asarray([[3, 1, 2], [6, 4, 5]])
for axis in (True, False, np.bool_(True), np.array(True), raw_pytorch.asarray(True)):
    try:
        raw_pytorch.argsort(values, axis=axis)
    except TypeError:
        pass
    else:
        raise AssertionError(f"accepted boolean argsort axis {axis!r}")

try:
    raw_pytorch.argsort(values, dim=True)
except TypeError:
    pass
else:
    raise AssertionError("accepted boolean argsort dim")

assert raw_pytorch.to_numpy(raw_pytorch.argsort(values, axis=1)).tolist() == [[1, 2, 0], [1, 2, 0]]
assert raw_pytorch.to_numpy(raw_pytorch.argsort(values, axis=raw_pytorch.asarray(1))).tolist() == [[1, 2, 0], [1, 2, 0]]
print("ok")
"""
    result = run_backend_code("numpy", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
