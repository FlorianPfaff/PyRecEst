import numpy as np
import numpy.testing as npt
import pytest
from pyrecest.distributions.hypertorus.fejer import apply_kernel_weights, fejer_weights


@pytest.mark.parametrize(
    "exponent",
    (
        np.nan,
        np.inf,
        -np.inf,
        True,
        1.0 + 0.0j,
        "1.0",
        np.array([1.0]),
    ),
)
def test_apply_kernel_weights_rejects_invalid_exponents(exponent):
    with pytest.raises(ValueError, match="exponent"):
        apply_kernel_weights(np.ones(3), exponent=exponent)


def test_apply_kernel_weights_accepts_zero_dimensional_real_exponent():
    result = apply_kernel_weights(np.ones(3), exponent=np.array(0.5))

    npt.assert_allclose(result, np.sqrt(fejer_weights(3)))
