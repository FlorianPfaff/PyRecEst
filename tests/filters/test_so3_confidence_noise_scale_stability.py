import unittest

import numpy as np
import numpy.testing as npt

from pyrecest.backend import array, to_numpy
from pyrecest.filters import SO3ProductParticleFilter


class SO3ConfidenceNoiseScaleStabilityTest(unittest.TestCase):
    def test_extreme_finite_scales_remain_finite(self):
        max_float = float(np.finfo(np.float64).max)
        min_sigma = max_float / 2.0

        sigma = SO3ProductParticleFilter.confidence_to_noise_std(
            array([1.0, 0.5, 0.0]),
            noise_std=min_sigma,
            max_noise_std=max_float,
        )
        sigma_np = np.asarray(to_numpy(sigma), dtype=float)

        self.assertTrue(np.isfinite(sigma_np).all())
        npt.assert_allclose(sigma_np[0], min_sigma, rtol=1e-15)
        npt.assert_allclose(sigma_np[2], max_float, rtol=1e-15)
        npt.assert_allclose(
            sigma_np[1] / max_float,
            np.sqrt(0.625),
            rtol=1e-15,
        )

    def test_confidence_one_preserves_minimum_when_scale_ratio_square_underflows(self):
        sigma = SO3ProductParticleFilter.confidence_to_noise_std(
            array([1.0]),
            noise_std=1.0,
            max_noise_std=1e200,
        )

        npt.assert_allclose(np.asarray(to_numpy(sigma), dtype=float), np.array([1.0]))


if __name__ == "__main__":
    unittest.main()
