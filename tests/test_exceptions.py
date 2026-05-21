import pytest
from pyrecest.exceptions import (
    BackendNotSupportedError,
    DimensionMismatchError,
    NumericalStabilityError,
    PyRecEstError,
    ShapeError,
)


def test_backend_not_supported_error_is_not_implemented_error():
    error = BackendNotSupportedError(
        "ExampleAPI.update",
        "jax",
        supported_backends=("numpy", "pytorch"),
        reason="uses in-place SciPy assignment",
    )

    assert isinstance(error, NotImplementedError)
    assert isinstance(error, PyRecEstError)
    assert "ExampleAPI.update" in str(error)
    assert "jax" in str(error)
    assert "numpy, pytorch" in str(error)


def test_shape_error_message_includes_expected_shape():
    error = ShapeError("measurement", (2, 3), expected="shape (n,)")
    assert isinstance(error, ValueError)
    assert "measurement" in str(error)
    assert "shape (n,)" in str(error)


def test_dimension_mismatch_error_records_dimensions():
    error = DimensionMismatchError("state", 4, "covariance", 3)
    assert error.left_dim == 4
    assert error.right_dim == 3
    assert "matching dimensions" in str(error)


def test_numerical_stability_error_message():
    with pytest.raises(NumericalStabilityError, match="Cholesky"):
        raise NumericalStabilityError(
            "Cholesky factorization", reason="matrix is not positive definite"
        )
