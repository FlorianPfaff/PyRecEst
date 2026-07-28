import numpy as np
import pyrecest.backend
import pytest
from pyrecest.sampling.leopardi_sampler import get_partition_points_cartesian

pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ == "jax",
    reason="Leopardi sampling uses SciPy root finding.",
)


@pytest.mark.parametrize("symmetry_type", ["plane", "antipodal"])
def test_small_symmetric_leopardi_partition_has_unique_points(symmetry_type):
    half = np.asarray(
        get_partition_points_cartesian(
            2,
            4,
            delete_half=True,
            symmetry_type=symmetry_type,
        )
    )
    full = np.asarray(
        get_partition_points_cartesian(
            2,
            4,
            delete_half=False,
            symmetry_type=symmetry_type,
        )
    )

    assert half.shape == (2, 3)
    assert full.shape == (4, 3)
    np.testing.assert_allclose(np.linalg.norm(half, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(full, axis=1), 1.0, atol=1e-12)
    assert np.unique(np.round(half, decimals=12), axis=0).shape[0] == 2
    assert np.unique(np.round(full, decimals=12), axis=0).shape[0] == 4
