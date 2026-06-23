import pytest
from tests.support.backend_runner import run_backend_code


def test_pytorch_choice_rejects_negative_probabilities_with_value_error():
    pytest.importorskip("torch")

    code = """
import pyrecest.backend as backend

try:
    backend.random.choice(3, p=[0.5, -0.1, 0.6])
except ValueError:
    pass
else:
    raise AssertionError("choice accepted a negative probability")
"""
    result = run_backend_code("pytorch", code)

    assert result.returncode == 0, result.stderr


def test_pytorch_choice_rejects_nonfinite_probabilities_with_value_error():
    pytest.importorskip("torch")

    code = """
import pyrecest.backend as backend

try:
    backend.random.choice(3, p=[0.5, float("nan"), 0.5])
except ValueError:
    pass
else:
    raise AssertionError("choice accepted a non-finite probability")
"""
    result = run_backend_code("pytorch", code)

    assert result.returncode == 0, result.stderr


def test_pytorch_choice_rejects_out_of_bounds_axis():
    pytest.importorskip("torch")

    code = """
import pyrecest.backend as backend

values = [[1, 2, 3], [4, 5, 6]]
for axis in (2, -3):
    try:
        backend.random.choice(values, axis=axis)
    except ValueError:
        pass
    else:
        raise AssertionError(f"choice accepted out-of-bounds axis {axis}")
"""
    result = run_backend_code("pytorch", code)

    assert result.returncode == 0, result.stderr
