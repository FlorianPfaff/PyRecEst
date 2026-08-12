import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, to_numpy
from pyrecest.distributions.nonperiodic.linear_box_particle_distribution import (
    LinearBoxParticleDistribution,
)


class LinearBoxParticleExtremeGeometryTest(unittest.TestCase):
    def test_centers_avoid_overflow_for_large_same_sign_bounds(self):
        backend_dtype = to_numpy(array([1.0])).dtype
        max_finite = np.finfo(backend_dtype).max
        lower_value = 0.75 * max_finite
        upper_value = max_finite
        dist = LinearBoxParticleDistribution(
            array([[lower_value]]), array([[upper_value]])
        )

        with np.errstate(over="raise", invalid="raise"):
            centers = to_numpy(dist.centers())

        expected = np.array([[0.875 * max_finite]], dtype=backend_dtype)
        self.assertTrue(np.isfinite(centers).all())
        npt.assert_allclose(centers, expected)

    def test_half_widths_avoid_overflow_for_opposite_sign_bounds(self):
        backend_dtype = to_numpy(array([1.0])).dtype
        max_finite = np.finfo(backend_dtype).max
        bound = 0.75 * max_finite
        dist = LinearBoxParticleDistribution(array([[-bound]]), array([[bound]]))

        with np.errstate(over="raise", invalid="raise"):
            half_widths = to_numpy(dist.half_widths())

        expected = np.array([[bound]], dtype=backend_dtype)
        self.assertTrue(np.isfinite(half_widths).all())
        npt.assert_allclose(half_widths, expected)


if __name__ == "__main__":
    unittest.main()
