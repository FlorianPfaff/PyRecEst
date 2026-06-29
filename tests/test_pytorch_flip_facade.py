import pytest

from tests.support.backend_runner import run_backend_code


def test_pytorch_flip_defaults_to_all_axes():
    pytest.importorskip("torch")

    code = """
import pyrecest.backend as backend

values = backend.asarray([[1, 2], [3, 4]])
flipped = backend.flip(values)
assert backend.to_numpy(flipped).tolist() == [[4, 3], [2, 1]]
"""

    result = run_backend_code("pytorch", code)

    assert result.returncode == 0, result.stderr
