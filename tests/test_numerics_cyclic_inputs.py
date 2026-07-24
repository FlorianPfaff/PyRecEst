import numpy as np
import pytest
from pyrecest.numerics import (
    assert_covariance_matrix,
    is_positive_semidefinite,
    is_symmetric,
    jittered_cholesky,
    nearest_symmetric_psd,
    symmetrize_matrix,
)


def test_covariance_helpers_reject_cyclic_object_matrix():
    matrix = np.empty((2, 2), dtype=object)
    matrix[:] = 0.0
    matrix[0, 0] = matrix

    assert not is_symmetric(matrix)
    assert not is_positive_semidefinite(matrix)

    with pytest.raises(ValueError, match="covariance must contain numeric values"):
        assert_covariance_matrix(matrix)
    with pytest.raises(ValueError, match="matrix must contain numeric values"):
        symmetrize_matrix(matrix)
    with pytest.raises(ValueError, match="matrix must contain numeric values"):
        nearest_symmetric_psd(matrix)
    with pytest.raises(ValueError, match="matrix must contain numeric values"):
        jittered_cholesky(matrix)
