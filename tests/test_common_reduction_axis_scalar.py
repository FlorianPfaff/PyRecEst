import numpy as np
from pyrecest._backend import _common


def test_numpy_scalar_integer_reduction_axis():
    assert _common._normalize_reduction_axes(np.asarray(0), 2) == (0,)
