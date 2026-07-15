import numpy as np
import pytest
from pyrecest._backend import _common


def test_common_size_rejects_boolean_axes():
    values = np.arange(6).reshape(2, 3)

    for axis in (True, False, np.bool_(True), np.bool_(False)):
        with pytest.raises(TypeError, match="an integer is required"):
            _common.size(values, axis=axis)


def test_common_size_accepts_integer_like_axes():
    values = np.arange(6).reshape(2, 3)

    assert _common.size(values, axis=0) == 2
    assert _common.size(values, axis=np.int64(1)) == 3
