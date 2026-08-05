import numpy as np
import pytest

from pyrecest.backend import linalg


@pytest.mark.parametrize(
    "exponent",
    (
        np.ma.array(0.5, mask=True),
        np.ma.masked,
    ),
)
def test_numpy_fractional_matrix_power_rejects_masked_exponent(exponent):
    with pytest.raises(TypeError, match="real scalar"):
        linalg.fractional_matrix_power(np.eye(2), exponent)
