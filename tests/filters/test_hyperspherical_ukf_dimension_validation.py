import numpy as np
import pyrecest.backend
import pytest

from pyrecest.filters.hyperspherical_ukf import HypersphericalUKF


pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ == "jax",
    reason="HypersphericalUKF is unsupported on JAX.",
)


@pytest.mark.parametrize(
    "invalid_dim",
    [
        0,
        -1,
        True,
        np.bool_(True),
        1.5,
        "2",
        np.array([2]),
        np.array(2.0),
    ],
)
def test_constructor_rejects_invalid_embedding_dimensions(invalid_dim):
    with pytest.raises(ValueError, match="dim must be a positive integer"):
        HypersphericalUKF(dim=invalid_dim)


def test_constructor_accepts_integer_scalar_wrappers():
    for dimension in (np.int64(3), np.array(3, dtype=np.int64)):
        filter_ = HypersphericalUKF(dim=dimension)

        assert filter_.filter_state.dim == 3
        assert filter_.get_point_estimate().shape == (3,)
