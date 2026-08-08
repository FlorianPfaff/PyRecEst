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


def test_numpy_fractional_matrix_power_accepts_clear_mask_exponent():
    result = linalg.fractional_matrix_power(
        np.diag([4.0, 9.0]),
        np.ma.array(0.5, mask=False),
    )

    np.testing.assert_allclose(result, np.diag([2.0, 3.0]))
