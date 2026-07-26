import numpy as np
import pytest

torch = pytest.importorskip("torch")

import pyrecest._backend.pytorch as pytorch_backend  # noqa: E402
import pyrecest.backend_tools  # noqa: E402,F401
from tests.support.backend_runner import run_backend_code  # noqa: E402


def _to_python(value):
    return pytorch_backend.to_numpy(value).tolist()


def test_raw_pytorch_flip_accepts_numpy_integer_axis():
    result = pytorch_backend.flip([[1, 2, 3], [4, 5, 6]], np.int64(1))

    assert _to_python(result) == [[3, 2, 1], [6, 5, 4]]


def test_raw_pytorch_flip_accepts_numpy_integer_axis_sequence():
    result = pytorch_backend.flip([[1, 2], [3, 4]], (np.int64(0), np.int64(1)))

    assert _to_python(result) == [[4, 3], [2, 1]]


def test_public_pytorch_flip_accepts_numpy_integer_axis():
    code = """
import numpy as np
import pyrecest.backend as backend

result = backend.flip([[1, 2, 3], [4, 5, 6]], np.int64(1))
assert backend.to_numpy(result).tolist() == [[3, 2, 1], [6, 5, 4]]
"""
    result = run_backend_code("pytorch", code)

    assert result.returncode == 0, result.stderr
