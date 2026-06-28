import numpy as np
import pytest
from pyrecest.distributions.so3_conversion import _validate_covariance_regularization


@pytest.mark.parametrize("value", [np.array(0.0), np.array(1e-6), np.float64(0.25)])
def test_covariance_regularization_accepts_numpy_scalar_values(value):
    assert _validate_covariance_regularization(value) == pytest.approx(
        float(np.asarray(value))
    )


@pytest.mark.parametrize(
    "value",
    [np.array([0.0]), np.array(True), np.array(-1.0), np.array(np.inf), "0.1"],
)
def test_covariance_regularization_rejects_non_numeric_nonfinite_or_nonscalar_values(
    value,
):
    with pytest.raises(ValueError, match="covariance_regularization"):
        _validate_covariance_regularization(value)
