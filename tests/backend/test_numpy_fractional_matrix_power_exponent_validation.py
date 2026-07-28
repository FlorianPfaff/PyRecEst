import numpy as np
import pytest
from pyrecest._backend.numpy import linalg

_MATRIX = np.diag([4.0, 9.0])


@pytest.mark.parametrize(
    "exponent",
    [
        True,
        np.bool_(False),
        np.array([0.5]),
        np.array([[0.5]]),
        np.timedelta64(2, "ns"),
        np.datetime64("1970-01-01T00:00:00.000000002"),
        np.array(np.timedelta64(2, "ns"), dtype=object),
        np.array(np.datetime64("1970-01-01T00:00:00.000000002"), dtype=object),
        "0.5",
        0.5 + 0.0j,
    ],
)
def test_fractional_matrix_power_rejects_non_real_scalar_exponents(exponent):
    with pytest.raises(TypeError, match="t must be a real scalar"):
        linalg.fractional_matrix_power(_MATRIX, exponent)


@pytest.mark.parametrize("exponent", [np.nan, np.inf, -np.inf, np.array(np.nan)])
def test_fractional_matrix_power_rejects_nonfinite_exponents(exponent):
    with pytest.raises(ValueError, match="t must be finite"):
        linalg.fractional_matrix_power(_MATRIX, exponent)


def test_fractional_matrix_power_accepts_zero_dimensional_real_exponent():
    result = linalg.fractional_matrix_power(_MATRIX, np.array(0.5))

    np.testing.assert_allclose(result, np.diag([2.0, 3.0]))


def test_fractional_matrix_power_validates_exponent_for_empty_batches():
    matrices = np.empty((0, 2, 2))

    with pytest.raises(TypeError, match="t must be a real scalar"):
        linalg.fractional_matrix_power(matrices, np.timedelta64(2, "ns"))
