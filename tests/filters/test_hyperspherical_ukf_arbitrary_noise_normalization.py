import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array
from pyrecest.filters.hyperspherical_ukf import HypersphericalUKF


class HypersphericalUKFArbitraryNoiseNormalizationTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Arbitrary-noise prediction is not supported on this backend",
    )
    def test_radial_model_scale_does_not_create_spurious_covariance(self):
        ukf = HypersphericalUKF(dim=2, alpha=1.0)
        noise_samples = np.array([[0.0, 1.0]])
        noise_weights = np.ones(2)

        def scaled_same_direction(_x, v):
            scale = 1.0 + float(np.asarray(v, dtype=float)[0])
            return array([scale, 0.0])

        ukf.predict_nonlinear_arbitrary_noise(
            scaled_same_direction, noise_samples, noise_weights
        )

        npt.assert_allclose(
            np.asarray(ukf.filter_state.mu, dtype=float),
            np.array([1.0, 0.0]),
            atol=1e-12,
        )
        npt.assert_allclose(
            np.asarray(ukf.filter_state.C, dtype=float),
            np.zeros((2, 2)),
            atol=1e-12,
        )
