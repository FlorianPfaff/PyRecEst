import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, diag
from pyrecest.distributions import EllipsoidalBallUniformDistribution


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="The stability regressions require extreme float64 values",
)
class TestEllipsoidalBallVolumeStability(unittest.TestCase):
    def test_small_positive_definite_shape_has_nonzero_representable_volume(self):
        axis_variance = 1e-200
        dist = EllipsoidalBallUniformDistribution(
            array([0.0, 0.0]),
            diag(array([axis_variance, axis_variance])),
        )

        expected_volume = np.pi * axis_variance
        npt.assert_allclose(
            dist.get_manifold_size(),
            expected_volume,
            rtol=1e-14,
            atol=0.0,
        )
        npt.assert_allclose(
            dist.pdf(array([0.0, 0.0])),
            1.0 / expected_volume,
            rtol=1e-14,
        )

    def test_balanced_extreme_axes_avoid_intermediate_product_overflow(self):
        large_variance = np.finfo(float).max / 4.0
        small_variance = 1.0 / large_variance
        dist = EllipsoidalBallUniformDistribution(
            array(np.zeros(6)),
            diag(
                array(
                    [
                        large_variance,
                        large_variance,
                        large_variance,
                        small_variance,
                        small_variance,
                        small_variance,
                    ]
                )
            ),
        )
        expected_volume = np.pi**3 / 6.0

        with np.errstate(over="raise", divide="raise", invalid="raise"):
            volume = dist.get_manifold_size()
            center_density = dist.pdf(array(np.zeros(6)))

        npt.assert_allclose(volume, expected_volume, rtol=5e-13)
        npt.assert_allclose(center_density, 1.0 / expected_volume, rtol=5e-13)


if __name__ == "__main__":
    unittest.main()
