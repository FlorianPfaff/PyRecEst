import numpy as np
import numpy.testing as npt
import pytest
from pyrecest.models.validation import (
    validate_covariance_matrix,
    validate_matrix,
    validate_vector,
)


def test_validation_accepts_zero_dimensional_numpy_integer_dimensions():
    vector = validate_vector([1.0, 2.0], dim=np.array(2, dtype=np.int64))
    matrix = validate_matrix(
        [[1.0, 2.0], [3.0, 4.0]],
        rows=np.array(2, dtype=np.int64),
        cols=np.array(2, dtype=np.int64),
    )
    covariance = validate_covariance_matrix(
        [[1.0, 0.0], [0.0, 2.0]],
        dim=np.array(2, dtype=np.int64),
    )

    npt.assert_allclose(np.asarray(vector), [1.0, 2.0])
    npt.assert_allclose(np.asarray(matrix), [[1.0, 2.0], [3.0, 4.0]])
    npt.assert_allclose(np.asarray(covariance), [[1.0, 0.0], [0.0, 2.0]])


@pytest.mark.parametrize(
    "dim",
    [np.array([2]), np.array(2.0), np.array(True), True, "2"],
)
def test_validate_vector_rejects_invalid_dimension_scalars(dim):
    with pytest.raises((TypeError, ValueError), match="dim"):
        validate_vector([1.0, 2.0], dim=dim)
