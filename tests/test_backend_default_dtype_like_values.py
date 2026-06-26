import numpy as np
import pytest
from pyrecest._backend import numpy as numpy_backend


def test_numpy_set_default_dtype_accepts_dtype_like_values():
    try:
        returned_dtype = numpy_backend.set_default_dtype(np.dtype("float32"))

        assert returned_dtype == np.dtype("float32")
        assert numpy_backend.get_default_dtype() == np.dtype("float32")
        assert numpy_backend.get_default_cdtype() == np.dtype("complex64")

        returned_dtype = numpy_backend.set_default_dtype(np.float64)

        assert returned_dtype == np.dtype("float64")
        assert numpy_backend.get_default_dtype() == np.dtype("float64")
        assert numpy_backend.get_default_cdtype() == np.dtype("complex128")
    finally:
        numpy_backend.set_default_dtype("float64")


def test_numpy_set_default_dtype_rejects_non_float_without_state_mutation():
    numpy_backend.set_default_dtype("float64")
    previous_dtype = numpy_backend.get_default_dtype()
    previous_cdtype = numpy_backend.get_default_cdtype()

    with pytest.raises(ValueError, match="float32 or float64"):
        numpy_backend.set_default_dtype(np.dtype("complex64"))

    assert numpy_backend.get_default_dtype() == previous_dtype
    assert numpy_backend.get_default_cdtype() == previous_cdtype
