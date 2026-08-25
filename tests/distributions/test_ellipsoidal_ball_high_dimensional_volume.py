import unittest

import mpmath
import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions import EllipsoidalBallUniformDistribution


class TestEllipsoidalBallHighDimensionalVolume(unittest.TestCase):
    def test_volume_and_pdf_remain_finite_after_gamma_overflow_threshold(self):
        dim = 342
        dist = EllipsoidalBallUniformDistribution(
            array(np.zeros(dim)), array(np.eye(dim))
        )

        with mpmath.workdps(80):
            expected_volume = float(
                mpmath.pi ** (dim / 2) / mpmath.gamma(dim / 2 + 1)
            )

        volume = dist.get_manifold_size()
        npt.assert_allclose(volume, expected_volume, rtol=1e-12, atol=0.0)
        npt.assert_allclose(
            dist.pdf(array(np.zeros(dim))),
            1.0 / expected_volume,
            rtol=1e-12,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
