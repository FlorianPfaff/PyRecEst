import numpy as np
import pytest
from pyrecest.numerics import (
    assert_covariance_matrix,
    is_positive_semidefinite,
    is_symmetric,
    jittered_cholesky,
    nearest_symmetric_psd,
)


def test_numerical_scalar_controls_reject_text_values():
    matrix = np.eye(2)

    with pytest.raises(ValueError, match="atol"):
        is_symmetric(matrix, atol="1e-9")
    with pytest.raises(ValueError, match="atol"):
        is_positive_semidefinite(matrix, atol="1e-9")
    with pytest.raises(ValueError, match="atol"):
        assert_covariance_matrix(matrix, atol="1e-9")
    with pytest.raises(ValueError, match="min_eigenvalue"):
        nearest_symmetric_psd(matrix, min_eigenvalue="0.0")
    with pytest.raises(ValueError, match="initial_jitter"):
        jittered_cholesky(matrix, initial_jitter="1e-12")
