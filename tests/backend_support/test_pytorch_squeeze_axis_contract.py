import importlib.util

import pytest

from tests.support.backend_runner import run_backend_code


def test_pytorch_backend_squeeze_rejects_non_singleton_axis():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest  # noqa: F401
import pyrecest.backend as backend

values = backend.array([[1], [2]])
for axis in (0, (0, 1)):
    try:
        backend.squeeze(values, axis=axis)
    except ValueError as exc:
        assert "size not equal to one" in str(exc), str(exc)
    else:
        raise AssertionError("expected non-singleton squeeze axis to raise")

result = backend.squeeze(values, axis=1)
assert tuple(result.shape) == (2,)
assert backend.to_numpy(result).tolist() == [1, 2]
print("ok")
"""

    result = run_backend_code("pytorch", code)

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_raw_pytorch_squeeze_rejects_non_singleton_axis_after_package_import():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest  # noqa: F401
import pyrecest._backend.pytorch as raw_pytorch

values = raw_pytorch.array([[1], [2]])
for axis in (0, (0, 1)):
    try:
        raw_pytorch.squeeze(values, axis=axis)
    except ValueError as exc:
        assert "size not equal to one" in str(exc), str(exc)
    else:
        raise AssertionError("expected non-singleton squeeze axis to raise")

result = raw_pytorch.squeeze(values, axis=1)
assert tuple(result.shape) == (2,)
assert raw_pytorch.to_numpy(result).tolist() == [1, 2]
print("ok")
"""

    result = run_backend_code("pytorch", code)

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
