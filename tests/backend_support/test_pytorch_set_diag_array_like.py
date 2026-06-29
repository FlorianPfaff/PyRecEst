import pytest

from tests.support.backend_runner import run_backend_code


def test_pytorch_set_diag_accepts_array_like_matrix_and_raw_entrypoint():
    pytest.importorskip("torch")

    code = """
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_backend

result = backend.set_diag([[1.0, 1.0], [1.0, 1.0]], [5.0, 6.0])
assert backend.to_numpy(result).tolist() == [[5.0, 1.0], [1.0, 6.0]]

raw_result = raw_backend.set_diag(
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [7.0, 8.0],
)
assert backend.to_numpy(raw_result).tolist() == [
    [7.0, 0.0, 0.0],
    [0.0, 8.0, 0.0],
]
"""
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr
