import numpy as np
import pytest
from pyrecest.numerics import (
    assert_covariance_matrix,
    is_positive_semidefinite,
    is_symmetric,
    jittered_cholesky,
    nearest_symmetric_psd,
)


@pytest.mark.parametrize(
    "temporal_scalar",
    [
        np.timedelta64(1, "ns"),
        np.datetime64("1970-01-01T00:00:00.000000001", "ns"),
        np.array(np.timedelta64(1, "ns")),
        np.array(np.datetime64("1970-01-01T00:00:00.000000001", "ns")),
        np.array(np.timedelta64(1, "ns"), dtype=object),
        np.array(
            np.datetime64("1970-01-01T00:00:00.000000001", "ns"),
            dtype=object,
        ),
    ],
)
def test_numerics_helpers_reject_temporal_scalar_controls(temporal_scalar):
    matrix = np.eye(2)

    with pytest.raises(ValueError, match="atol"):
        is_symmetric(matrix, atol=temporal_scalar)
    with pytest.raises(ValueError, match="atol"):
        is_positive_semidefinite(matrix, atol=temporal_scalar)
    with pytest.raises(ValueError, match="atol"):
        assert_covariance_matrix(matrix, atol=temporal_scalar)
    with pytest.raises(ValueError, match="min_eigenvalue"):
        nearest_symmetric_psd(matrix, min_eigenvalue=temporal_scalar)
    with pytest.raises(ValueError, match="initial_jitter"):
        jittered_cholesky(matrix, initial_jitter=temporal_scalar)
    with pytest.raises(ValueError, match="max_attempts"):
        jittered_cholesky(matrix, max_attempts=temporal_scalar)
    with pytest.raises(ValueError, match="dim"):
        assert_covariance_matrix(matrix, dim=temporal_scalar)
