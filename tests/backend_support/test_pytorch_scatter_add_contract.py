import importlib.util

import pyrecest.backend as backend
import pytest
from tests.support.backend_runner import run_backend_code


def _to_python(value):
    converted = backend.to_numpy(value)
    return converted.tolist() if hasattr(converted, "tolist") else converted


@pytest.mark.backend_portable
def test_pytorch_scatter_add_accepts_array_like_inputs():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific backend contract")

    result = backend.scatter_add([10, 20, 30], 0, [0, 2], [1, 2])

    assert _to_python(result) == [11, 20, 32]


@pytest.mark.backend_portable
def test_pytorch_scatter_add_rejects_boolean_dim():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific backend contract")

    with pytest.raises(TypeError):
        backend.scatter_add([1, 2], True, [0], [1])


@pytest.mark.backend_portable
def test_pytorch_scatter_add_rejects_non_integer_indices():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific backend contract")

    np = pytest.importorskip("numpy")
    for bad_index in ([0.0], [True], np.array([0.0])):
        with pytest.raises(TypeError):
            backend.scatter_add([1, 2], 0, bad_index, [1])

    empty_result = backend.scatter_add([1, 2], 0, [], [])
    assert _to_python(empty_result) == [1, 2]


@pytest.mark.backend_portable
def test_raw_pytorch_scatter_add_rejects_non_integer_indices_with_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "numpy",
        """
import numpy as np
import pyrecest  # noqa: F401  # triggers raw-backend compatibility patches
import pyrecest._backend.pytorch as raw_pytorch

for bad_index in ([0.0], [True], np.array([0.0])):
    try:
        raw_pytorch.scatter_add([1, 2], 0, bad_index, [1])
    except TypeError:
        pass
    else:
        raise AssertionError("scatter_add accepted non-integer indices")

empty_result = raw_pytorch.scatter_add([1, 2], 0, [], [])
assert raw_pytorch.to_numpy(empty_result).tolist() == [1, 2]
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
