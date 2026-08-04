import unittest

import numpy as np
import numpy.testing as npt

import pyrecest.backend
from pyrecest.backend import array, cos, eye, sin
from pyrecest.smoothers import SO3ChordalMeanSmoother


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Extreme float64 regression is specific to the NumPy backend",
)
class SO3ChordalMeanWeightStabilityTest(unittest.TestCase):
    @staticmethod
    def _z_rotation(angle):
        return array(
            [
                [cos(angle), -sin(angle), 0.0],
                [sin(angle), cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    def test_chordal_mean_preserves_extreme_finite_weight_ratio(self):
        max_float = np.finfo(float).max
        identity = eye(3)
        quarter_turn = self._z_rotation(0.5 * np.pi)

        mean_rotation = SO3ChordalMeanSmoother.chordal_mean(
            [identity, quarter_turn],
            weights=array([max_float, 0.5 * max_float]),
        )

        npt.assert_allclose(
            mean_rotation,
            self._z_rotation(np.arctan(0.5)),
            atol=1.0e-6,
        )

    def test_smoothing_scales_sample_and_kernel_weights_before_products(self):
        max_float = np.finfo(float).max
        identity = eye(3)
        quarter_turn = self._z_rotation(0.5 * np.pi)
        smoother = SO3ChordalMeanSmoother(
            window_size=3,
            kernel_weights=array([max_float, max_float, max_float]),
        )

        smoothed = smoother.smooth(
            [identity, quarter_turn],
            weights=array([max_float, 0.5 * max_float]),
        )

        expected = self._z_rotation(np.arctan(0.5))
        self.assertEqual(len(smoothed), 2)
        npt.assert_allclose(smoothed[0], expected, atol=1.0e-6)
        npt.assert_allclose(smoothed[1], expected, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
