import pytest
from tests.support.backend_runner import run_backend_code

_ARRAY_DTYPE_NONE_CODE = """
import pyrecest.backend as backend

backend.set_default_dtype("float32")
try:
    omitted_real = backend.array([1.0])
    explicit_none_real = backend.array([1.0], dtype=None)
    positional_none_real = backend.array([1.0], None)
    explicit_none_complex = backend.array([1.0 + 2.0j], dtype=None)
    explicit_none_integer = backend.array([1], dtype=None)

    assert str(omitted_real.dtype) == "float32", omitted_real.dtype
    assert str(explicit_none_real.dtype) == "float32", explicit_none_real.dtype
    assert str(positional_none_real.dtype) == "float32", positional_none_real.dtype
    assert str(explicit_none_complex.dtype) == "complex64", explicit_none_complex.dtype
    assert str(explicit_none_integer.dtype).startswith("int"), explicit_none_integer.dtype
finally:
    backend.set_default_dtype("float64")
"""


@pytest.mark.parametrize("backend_name", ["numpy", "autograd"])
def test_shared_numpy_array_dtype_none_uses_default_dtype(backend_name):
    if backend_name == "autograd":
        pytest.importorskip("autograd")

    result = run_backend_code(backend_name, _ARRAY_DTYPE_NONE_CODE)

    assert result.returncode == 0, result.stderr
