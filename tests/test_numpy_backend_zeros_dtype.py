import importlib.util

import numpy as np
import pytest
from pyrecest._backend import numpy as numpy_backend
from tests.support.backend_runner import run_backend_code


def test_numpy_zeros_preserves_explicit_positional_dtype():
    result = numpy_backend.zeros((2,), np.float32)

    assert result.dtype == np.dtype(np.float32)


def test_numpy_zeros_uses_default_dtype_when_dtype_is_omitted():
    result = numpy_backend.zeros((2,))

    assert result.dtype == np.dtype(numpy_backend.get_default_dtype())


@pytest.mark.backend_portable
def test_numpy_zeros_uses_numpy_dtype_after_pytorch_backend_initialization():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import numpy as np
import pyrecest.backend as backend
from pyrecest._backend import numpy as numpy_backend

result = numpy_backend.zeros((2,))
assert result.dtype == np.dtype("float64"), (
    backend.__backend_name__,
    backend.get_default_dtype(),
    numpy_backend.get_default_dtype(),
    result.dtype,
)
assert numpy_backend.get_default_dtype() == np.dtype("float64")
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
