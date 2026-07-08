import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_complex_positive_definite_predicate_returns_python_bool():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend

matrix = backend.array(
    [[2.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 3.0 + 0.0j]]
)
value = backend.linalg.is_single_matrix_pd(matrix)
assert isinstance(value, bool), type(value)
assert value is True
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.backend_portable
def test_pytorch_linalg_norm_rejects_boolean_sequence_axes():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import numpy as np
import pyrecest.backend as backend

values = backend.array([[3.0, 4.0], [5.0, 12.0]])
expected = backend.array([5.0, 13.0])

for axis in [
    True,
    False,
    np.bool_(True),
    np.array(True),
    backend.asarray(True),
    [True],
    (True,),
    [np.bool_(False)],
    (np.array(True),),
    [backend.asarray(True)],
]:
    try:
        backend.linalg.norm(values, axis=axis)
    except TypeError:
        pass
    else:
        raise AssertionError(f"accepted boolean norm axis {axis!r}")

for axis in [
    1,
    np.int64(1),
    np.array(1),
    backend.asarray(1),
    [1],
    (1,),
    np.array([1]),
    backend.asarray([1]),
]:
    result = backend.linalg.norm(values, axis=axis)
    assert backend.allclose(result, expected), axis

print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
