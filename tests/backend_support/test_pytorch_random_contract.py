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
