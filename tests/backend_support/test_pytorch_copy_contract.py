import numpy as np
import pytest
from tests.support.backend_runner import run_backend_code


def test_pytorch_copy_returns_backend_tensors_for_array_like_inputs():
    pytest.importorskip("torch")

    import pyrecest  # noqa: F401
    import pyrecest._backend.pytorch as raw_pytorch

    scalar_copy = raw_pytorch.copy(1.5)
    assert raw_pytorch.is_array(scalar_copy)
    assert tuple(scalar_copy.shape) == ()
    assert float(scalar_copy) == 1.5

    sequence_copy = raw_pytorch.copy([[1.0, 2.0], [3.0, 4.0]])
    assert raw_pytorch.is_array(sequence_copy)
    assert sequence_copy.tolist() == [[1.0, 2.0], [3.0, 4.0]]

    source = np.array([1.0, 2.0])
    array_copy = raw_pytorch.copy(source)
    source[0] = 99.0
    assert raw_pytorch.is_array(array_copy)
    assert array_copy.tolist() == [1.0, 2.0]


def test_public_pytorch_copy_returns_backend_tensor_for_array_like_inputs():
    pytest.importorskip("torch")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend

copied = backend.copy([[1.0, 2.0], [3.0, 4.0]])
assert backend.is_array(copied)
assert copied.tolist() == [[1.0, 2.0], [3.0, 4.0]]
""",
    )
    assert result.returncode == 0, result.stderr
