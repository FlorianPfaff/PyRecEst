import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_array_accepts_non_native_numpy_dtype_aliases():
    pytest.importorskip("torch")

    result = run_backend_code(
        "pytorch",
        """
import numpy as np
import pyrecest.backend as backend

values = backend.array([1, 2], dtype=np.dtype(">f8"))
assert values.dtype == backend.float64
assert values.tolist() == [1.0, 2.0]

ints = backend.array([1, 2], dtype=np.dtype(">i4"))
assert ints.dtype == backend.int32
assert ints.tolist() == [1, 2]

cast_values = backend.cast(backend.array([1, 2]), np.dtype(">f8"))
assert cast_values.dtype == backend.float64
assert cast_values.tolist() == [1.0, 2.0]
""",
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.backend_portable
def test_raw_pytorch_array_accepts_non_native_numpy_dtype_aliases():
    pytest.importorskip("torch")

    result = run_backend_code(
        "numpy",
        """
import numpy as np
import pyrecest._backend.pytorch as raw_pytorch

values = raw_pytorch.array([1, 2], dtype=np.dtype(">f8"))
assert values.dtype == raw_pytorch.float64
assert values.tolist() == [1.0, 2.0]

ints = raw_pytorch.array([1, 2], dtype=np.dtype(">i4"))
assert ints.dtype == raw_pytorch.int32
assert ints.tolist() == [1, 2]

cast_values = raw_pytorch.cast(raw_pytorch.array([1, 2]), np.dtype(">f8"))
assert cast_values.dtype == raw_pytorch.float64
assert cast_values.tolist() == [1.0, 2.0]
""",
    )

    assert result.returncode == 0, result.stderr
