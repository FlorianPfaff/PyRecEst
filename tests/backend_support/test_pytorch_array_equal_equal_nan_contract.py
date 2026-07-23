import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

_ARRAY_EQUAL_EQUAL_NAN_SCRIPT = """
import importlib

raw_pytorch = importlib.import_module("pyrecest._backend.pytorch")

assert raw_pytorch.array_equal(
    [1.0, float("nan")],
    raw_pytorch.asarray([1.0, float("nan")]),
    equal_nan=True,
)
assert not raw_pytorch.array_equal([1.0, float("nan")], [1.0, float("nan")])
assert not raw_pytorch.array_equal(
    [1.0, float("nan")],
    [1.0, 2.0],
    equal_nan=True,
)
assert raw_pytorch.array_equal(
    [complex(float("nan"), 1.0)],
    [complex(float("nan"), 2.0)],
    equal_nan=True,
)
"""


@pytest.mark.backend_portable
@pytest.mark.parametrize("backend_name", ["numpy", "pytorch"])
def test_raw_pytorch_array_equal_accepts_equal_nan_keyword(backend_name):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(backend_name, _ARRAY_EQUAL_EQUAL_NAN_SCRIPT)

    assert result.returncode == 0, result.stderr


@pytest.mark.backend_portable
def test_public_pytorch_array_equal_accepts_equal_nan_keyword_when_active():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend

assert backend.array_equal([1.0, float("nan")], [1.0, float("nan")], equal_nan=True)
assert not backend.array_equal([1.0, float("nan")], [1.0, 2.0], equal_nan=True)
""",
    )

    assert result.returncode == 0, result.stderr
