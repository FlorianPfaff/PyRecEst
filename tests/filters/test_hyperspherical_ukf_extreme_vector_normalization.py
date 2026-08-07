import numpy as np
import numpy.testing as npt
import pyrecest.backend
import pytest
from pyrecest.backend import array
from pyrecest.filters.hyperspherical_ukf import HypersphericalUKF


@pytest.mark.skipif(
    pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
    reason="Arbitrary-noise prediction is not supported on this backend",
)
def test_arbitrary_noise_prediction_normalizes_extreme_finite_vectors():
    """Finite transition vectors must not collapse when their norm overflows."""
    filter_instance = HypersphericalUKF(dim=2)
    largest_float = np.finfo(np.float64).max

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        filter_instance.predict_nonlinear_arbitrary_noise(
            lambda _state, _noise: array([largest_float, largest_float]),
            noise_samples=np.zeros((1, 1)),
            noise_weights=np.ones(1),
        )

    npt.assert_allclose(
        np.asarray(filter_instance.get_point_estimate(), dtype=float),
        np.full(2, 1.0 / np.sqrt(2.0)),
        atol=1e-12,
    )
