import numpy as np
from pyrecest._backend import _backend_config
from pyrecest._backend import numpy as numpy_backend


def test_dynamic_dtype_preserves_explicit_positional_dtype_for_zeros():
    result = numpy_backend.zeros((2,), np.float32)

    assert result.dtype == np.dtype(np.float32)


def test_dynamic_dtype_preserves_explicit_positional_dtype_for_empty():
    result = numpy_backend.empty((2,), np.float32)

    assert result.dtype == np.dtype(np.float32)


def test_dynamic_dtype_preserves_explicit_positional_dtype_for_linspace():
    result = numpy_backend.linspace(0.0, 1.0, 3, True, False, np.float32)

    assert result.dtype == np.dtype(np.float32)


def test_dynamic_dtype_uses_default_dtype_for_omitted_or_none_dtype():
    previous_dtype = _backend_config.DEFAULT_DTYPE
    expected_dtype = np.dtype(np.float64)
    try:
        _backend_config.DEFAULT_DTYPE = expected_dtype
        assert numpy_backend.zeros((2,)).dtype == expected_dtype
        assert numpy_backend.empty((2,), None).dtype == expected_dtype
        assert (
            numpy_backend.linspace(0.0, 1.0, 3, True, False, None).dtype
            == expected_dtype
        )
    finally:
        _backend_config.DEFAULT_DTYPE = previous_dtype
