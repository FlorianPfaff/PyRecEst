import pytest
from tests.support.backend_runner import run_backend_code


def test_pytorch_backend_imports_and_exposes_diagonal_trace():
    pytest.importorskip("torch")

    code = """
import pyrecest.backend as backend

values = backend.asarray([
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
])
diag = backend.diagonal(values, offset=1, axis1=1, axis2=2)
trace = backend.trace(values, offset=1, axis1=1, axis2=2)

assert backend.__backend_name__ == "pytorch"
assert backend.to_numpy(diag).tolist() == [[2.0, 6.0], [8.0, 12.0]]
assert backend.to_numpy(trace).tolist() == [8.0, 20.0]
"""

    result = run_backend_code("pytorch", code)

    assert result.returncode == 0, result.stderr
