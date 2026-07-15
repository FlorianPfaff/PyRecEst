import numpy as np
import pytest
from pyrecest.exceptions import (
    DimensionMismatchError,
    NumericalStabilityError,
    ShapeError,
)
from pyrecest.numerics import (
    assert_covariance_matrix,
    is_positive_semidefinite,
    is_symmetric,
    jittered_cholesky,
    nearest_symmetric_psd,
    symmetrize_matrix,
)


class UncoercibleArray:
    def __array__(self, dtype=None):
        del dtype
        raise RuntimeError("cannot convert")


class OverflowingArray:
    def __array__(self, dtype=None):
        del dtype
        raise OverflowError("cannot convert")


def test_symmetrize_matrix_and_psd_projection():
    matrix = np.array([[1.0, 2.0], [0.0, -0.1]])
    symmetric = np.asarray(symmetrize_matrix(matrix))
    assert np.allclose(symmetric, symmetric.T)

    repaired = np.asarray(nearest_symmetric_psd(matrix))
    assert is_symmetric(repaired)
    assert is_positive_semidefinite(repaired)


def test_empty_square_matrix_is_symmetric_psd_covariance():
    matrix = np.empty((0, 0))

    assert is_symmetric(matrix)
    assert is_positive_semidefinite(matrix)
    validated = np.asarray(assert_covariance_matrix(matrix, dim=0))
    assert validated.shape == (0, 0)


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([[True, False], [False, True]]),
        np.array([["1.0", "0.0"], ["0.0", "1.0"]]),
        np.array([[1.0, False], [0.0, 1.0]], dtype=object),
        np.array([[None, 0.0], [0.0, 1.0]], dtype=object),
        np.array(
            [
                [np.datetime64("2020-01-01"), np.datetime64("2020-01-02")],
                [np.datetime64("2020-01-02"), np.datetime64("2020-01-03")],
            ]
        ),
        np.array(
            [
                [np.timedelta64(1, "D"), np.timedelta64(0, "D")],
                [np.timedelta64(0, "D"), np.timedelta64(1, "D")],
            ]
        ),
        np.array([[np.datetime64("2020-01-01"), 0.0], [0.0, 1.0]], dtype=object),
        np.array([[np.array(True), 0.0], [0.0, 1.0]], dtype=object),
        np.array([[np.array("1.0"), 0.0], [0.0, 1.0]], dtype=object),
        np.array([[np.array(1.0 + 2.0j), 0.0], [0.0, 1.0]], dtype=object),
    ],
)
def test_covariance_helpers_reject_bool_text_temporal_and_none_matrices(matrix):
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


@pytest.mark.parametrize("array_like_factory", [UncoercibleArray, OverflowingArray])
def test_covariance_helpers_reject_uncoercible_array_like_inputs(array_like_factory):
    matrix = array_like_factory()

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


@pytest.mark.parametrize("array_like_factory", [UncoercibleArray, OverflowingArray])
def test_covariance_helpers_reject_uncoercible_scalar_controls(array_like_factory):
    value = array_like_factory()
    matrix = np.eye(2)

    with pytest.raises(ValueError, match="atol"):
        is_symmetric(matrix, atol=value)
    with pytest.raises(ValueError, match="atol"):
        is_positive_semidefinite(matrix, atol=value)
    with pytest.raises(ValueError, match="min_eigenvalue"):
        nearest_symmetric_psd(matrix, min_eigenvalue=value)
    with pytest.raises(ValueError, match="initial_jitter"):
        jittered_cholesky(matrix, initial_jitter=value)
    with pytest.raises(ValueError, match="max_attempts"):
        jittered_cholesky(matrix, max_attempts=value)
    with pytest.raises(ValueError, match="dim"):
        assert_covariance_matrix(matrix, dim=value)


def test_covariance_helpers_reject_nonfinite_matrices():
    matrix = np.array([[np.inf, 0.0], [0.0, 1.0]])

    assert not is_symmetric(matrix)
    assert not is_positive_semidefinite(matrix)

    with pytest.raises(NumericalStabilityError, match="finite"):
        assert_covariance_matrix(matrix)
    with pytest.raises(NumericalStabilityError, match="finite"):
        nearest_symmetric_psd(matrix)
    with pytest.raises(NumericalStabilityError, match="finite"):
        jittered_cholesky(matrix)


def test_matrix_repair_helpers_reject_non_square_matrices_with_shape_error():
    matrix = np.ones((2, 3))

    with pytest.raises(ShapeError, match="square matrix"):
        symmetrize_matrix(matrix)
    with pytest.raises(ShapeError, match="square matrix"):
        nearest_symmetric_psd(matrix)
    with pytest.raises(ShapeError, match="square matrix"):
        jittered_cholesky(matrix)


def test_nearest_symmetric_psd_rejects_invalid_min_eigenvalue():
    matrix = np.eye(2)

    for min_eigenvalue in (-1.0, np.nan, np.inf, True, np.array([0.0])):
        with pytest.raises(ValueError, match="min_eigenvalue"):
            nearest_symmetric_psd(matrix, min_eigenvalue=min_eigenvalue)


def test_jittered_cholesky_reports_jitter():
    matrix = np.array([[1.0, 0.0], [0.0, 0.0]])
    factor, jitter = jittered_cholesky(matrix)
    assert np.asarray(factor).shape == (2, 2)
    assert jitter > 0.0


def test_jittered_cholesky_rejects_invalid_retry_controls():
    matrix = np.eye(2)

    for initial_jitter in (0.0, -1e-12, np.nan, np.inf, True, np.array([1e-12])):
        with pytest.raises(ValueError, match="initial_jitter"):
            jittered_cholesky(matrix, initial_jitter=initial_jitter)

    for max_attempts in (-1, 1.5, True, np.array([1])):
        with pytest.raises(ValueError, match="max_attempts"):
            jittered_cholesky(matrix, max_attempts=max_attempts)


def test_jittered_cholesky_accepts_numpy_integer_retry_count():
    matrix = np.array([[1.0, 0.0], [0.0, 0.0]])
    _, jitter = jittered_cholesky(matrix, max_attempts=np.int64(3))

    assert jitter > 0.0


def test_assert_covariance_matrix_validates_dimension_argument():
    matrix = np.eye(2)

    np.testing.assert_array_equal(
        np.asarray(assert_covariance_matrix(matrix, dim=2)), matrix
    )

    with pytest.raises(
        DimensionMismatchError, match="covariance has dimension 2"
    ) as exc_info:
        assert_covariance_matrix(matrix, dim=3)
    error = exc_info.value
    assert error.left_name == "covariance"
    assert error.left_dim == 2
    assert error.right_name == "expected"
    assert error.right_dim == 3

    for invalid_dim in (-1, 2.0, True, np.array([2])):
        with pytest.raises(ValueError, match="dim"):
            assert_covariance_matrix(matrix, dim=invalid_dim)


def test_assert_covariance_matrix_rejects_non_psd():
    with pytest.raises(NumericalStabilityError):
        assert_covariance_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]))
