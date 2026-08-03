import numpy as np
from pyrecest.numerics import (
    jittered_cholesky,
    nearest_symmetric_psd,
    symmetrize_matrix,
)


def test_symmetrize_matrix_avoids_finite_overflow_and_tiny_underflow():
    maximum = np.finfo(float).max
    matrix = np.array([[maximum, maximum], [maximum / 2.0, maximum / 2.0]])

    with np.errstate(over="raise", invalid="raise"):
        symmetric = np.asarray(symmetrize_matrix(matrix))

    expected = np.array([[maximum, maximum * 0.75], [maximum * 0.75, maximum / 2.0]])
    np.testing.assert_array_equal(symmetric, expected)
    np.testing.assert_array_equal(symmetric, symmetric.T)

    smallest = np.nextafter(0.0, 1.0)
    tiny = np.full((2, 2), smallest)
    np.testing.assert_array_equal(np.asarray(symmetrize_matrix(tiny)), tiny)


def test_nearest_symmetric_psd_preserves_maximum_scale_diagonal():
    matrix = np.eye(2) * np.finfo(float).max

    with np.errstate(over="raise", invalid="raise"):
        repaired = np.asarray(nearest_symmetric_psd(matrix))

    assert np.all(np.isfinite(repaired))
    np.testing.assert_allclose(
        repaired,
        matrix,
        rtol=4.0 * np.finfo(float).eps,
        atol=0.0,
    )


def test_jittered_cholesky_handles_maximum_scale_diagonal_without_jitter():
    matrix = np.eye(2) * np.finfo(float).max

    with np.errstate(over="raise", invalid="raise"):
        factor, jitter = jittered_cholesky(matrix)

    factor = np.asarray(factor)
    assert jitter == 0.0
    assert np.all(np.isfinite(factor))
    np.testing.assert_allclose(
        factor @ factor.T,
        matrix,
        rtol=4.0 * np.finfo(float).eps,
        atol=0.0,
    )
