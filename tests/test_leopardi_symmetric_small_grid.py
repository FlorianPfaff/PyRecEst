import numpy as np
import pyrecest.backend as backend
import pytest
from pyrecest.sampling.leopardi_sampler import (
    get_equal_area_caps,
    get_partition_points_cartesian,
)

pytestmark = pytest.mark.skipif(
    getattr(backend, "__backend_name__", "numpy") == "jax",
    reason="Leopardi sampling is not supported on the JAX backend",
)


def _as_numpy(value):
    to_numpy = getattr(backend, "to_numpy", None)
    if callable(to_numpy):
        value = to_numpy(value)
    return np.asarray(value)


def test_symmetric_four_point_partition_uses_two_collars():
    _, n_regions = get_equal_area_caps(2, 4, symmetric=True)

    assert _as_numpy(n_regions).tolist() == [1, 1, 1, 1]


def test_symmetric_four_point_northern_half_has_distinct_points():
    north = _as_numpy(
        get_partition_points_cartesian(
            2,
            4,
            delete_half=True,
            symmetry_type="",
        )
    )

    assert north.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(north, axis=1), 1.0, atol=1e-12)
    assert np.unique(np.round(north, decimals=12), axis=0).shape[0] == 2
    assert np.all(north[:, -1] >= -1e-12)
