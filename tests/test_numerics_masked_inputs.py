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


def test_covariance_helpers_reject_masked_matrix_entries():
    matrix = np.ma.array(
        [[1.0, 0.0], [0.0, 1.0]],
        mask=[[False, True], [False, False]],
    )

    assert not is_symmetric(matrix)
    assert not is_positive_semidefinite(matrix)

    for helper in (
        assert_covariance_matrix,
        symmetrize_matrix,
        nearest_symmetric_psd,
        jittered_cholesky,
    ):
        with pytest.raises(ValueError, match="must contain numeric values"):
            helper(matrix)


def test_covariance_helpers_reject_masked_scalar_controls():
    matrix = np.eye(2)

    invalid_calls = (
        lambda: is_symmetric(matrix, atol=np.ma.array(1e-10, mask=True)),
        lambda: is_positive_semidefinite(matrix, atol=np.ma.array(1e-10, mask=True)),
        lambda: nearest_symmetric_psd(
            matrix, min_eigenvalue=np.ma.array(0.0, mask=True)
        ),
        lambda: jittered_cholesky(matrix, initial_jitter=np.ma.array(1e-12, mask=True)),
        lambda: jittered_cholesky(matrix, max_attempts=np.ma.array(2, mask=True)),
        lambda: assert_covariance_matrix(matrix, dim=np.ma.array(2, mask=True)),
    )

    for call in invalid_calls:
        with pytest.raises(ValueError):
            call()


def test_covariance_helpers_accept_unmasked_masked_arrays():
    matrix = np.ma.array(np.eye(2), mask=False)

    assert is_symmetric(matrix, atol=np.ma.array(1e-10, mask=False))
    assert is_positive_semidefinite(matrix, atol=np.ma.array(1e-10, mask=False))
    np.testing.assert_array_equal(
        np.asarray(
            assert_covariance_matrix(
                matrix,
                dim=np.ma.array(2, mask=False),
            )
        ),
        np.eye(2),
    )
    factor, jitter = jittered_cholesky(
        matrix,
        initial_jitter=np.ma.array(1e-12, mask=False),
        max_attempts=np.ma.array(2, mask=False),
    )
    np.testing.assert_array_equal(np.asarray(factor), np.eye(2))
    assert jitter == 0.0
