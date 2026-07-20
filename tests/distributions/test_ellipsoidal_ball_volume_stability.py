import unittest

import numpy as np
import numpy.testing as npt

import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, diag
from pyrecest.distributions import EllipsoidalBallUniformDistribution


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="The underflow regression requires float64 values below float32 range",
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


if __name__ == "__main__":
    unittest.main()
