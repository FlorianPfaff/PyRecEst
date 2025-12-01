"""Test for uniform distribution for hyperhemispheres"""

import unittest

import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, linalg, ones, pi, random, reshape
from pyrecest.distributions import HyperhemisphericalUniformDistribution


def get_random_points(n, d):
    random.seed(10)
    points = random.normal(size=(n, d + 1))
    points = points[points[:, -1] >= 0, :]
    points /= reshape(linalg.norm(points, axis=1), (-1, 1))
    return points


class TestHyperhemisphericalUniformDistribution(unittest.TestCase):
    """Test for uniform distribution for hyperhemispheres"""

    def test_pdf_2d(self):
        hhud = HyperhemisphericalUniformDistribution(2)

        points = get_random_points(100, 2)

        self.assertTrue(
            allclose(hhud.pdf(points), ones(points.shape[0]) / (2 * pi), atol=1e-6)
        )

    def test_pdf_3d(self):
        hhud = HyperhemisphericalUniformDistribution(3)

        points = get_random_points(100, 3)
        # jscpd:ignore-start
        self.assertTrue(
            allclose(hhud.pdf(points), ones(points.shape[0]) / (pi**2), atol=1e-6)
        )
        # jscpd:ignore-end

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    def test_integrate_S2(self):
        hhud = HyperhemisphericalUniformDistribution(2)
        self.assertAlmostEqual(hhud.integrate(), 1.0, delta=1e-6)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    def test_integrate_S3(self):
        hhud = HyperhemisphericalUniformDistribution(3)
        self.assertAlmostEqual(hhud.integrate(), 1, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
