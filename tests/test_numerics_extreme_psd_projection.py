import numpy as np

from pyrecest.numerics import nearest_symmetric_psd


def test_nearest_symmetric_psd_scales_dense_extreme_covariance_before_eigh():
    maximum = np.finfo(float).max
    matrix = np.full((2, 2), maximum)

    with np.errstate(over="raise", invalid="raise"):
        repaired = np.asarray(nearest_symmetric_psd(matrix))

    assert np.all(np.isfinite(repaired))
    np.testing.assert_allclose(
        repaired,
        matrix,
        rtol=8.0 * np.finfo(float).eps,
        atol=0.0,
    )
