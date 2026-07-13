import unittest

import numpy as np
import numpy.testing as npt

import pyrecest.backend
from pyrecest.backend import array
from pyrecest.filters.hyperspherical_ukf import HypersphericalUKF


@unittest.skipIf(
    pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
    reason="Arbitrary-noise prediction is not supported on this backend",
)
class HypersphericalUKFArbitraryNoiseNormalizationTest(unittest.TestCase):
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

    def test_maximum_finite_weights_do_not_overflow(self):
        ukf = HypersphericalUKF(dim=2, alpha=1.0)
        noise_samples = np.array([[0.0, 1.0]])
        noise_weights = np.full(2, np.finfo(float).max)

        def fixed_direction(_x, _v):
            return array([1.0, 0.0])

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            ukf.predict_nonlinear_arbitrary_noise(
                fixed_direction, noise_samples, noise_weights
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

    def test_rejects_empty_noise_support(self):
        ukf = HypersphericalUKF(dim=2, alpha=1.0)

        with self.assertRaisesRegex(ValueError, "at least one sample"):
            ukf.predict_nonlinear_arbitrary_noise(
                lambda x, _v: x,
                np.empty((1, 0)),
                np.empty((0,)),
            )
