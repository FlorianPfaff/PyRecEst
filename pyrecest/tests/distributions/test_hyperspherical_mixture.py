import unittest

from numpy.testing import assert_allclose

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import arange, array, linspace, meshgrid, pi, sqrt, stack, sum
from pyrecest.distributions import (
    AbstractHypersphereSubsetDistribution,
    HypersphericalMixture,
    VonMisesFisherDistribution,
    WatsonDistribution,
)


class HypersphericalMixtureTest(unittest.TestCase):
    def test_pdf_3d(self):
        wad = WatsonDistribution(array([0.0, 0.0, 1.0]), -10.0)
        vmf = VonMisesFisherDistribution(array([0.0, 0.0, 1.0]), 1.0)
        w = array([0.3, 0.7])
        smix = HypersphericalMixture([wad, vmf], w)

        phi, theta = meshgrid(
            linspace(0.0, 2.0 * pi, 10), linspace(-pi / 2.0, pi / 2.0, 10)
        )
        points = AbstractHypersphereSubsetDistribution.hypersph_to_cart(
            stack([phi.ravel(), theta.ravel()], axis=-1)
        )

        assert_allclose(
            smix.pdf(points),
            w[0] * wad.pdf(points) + w[1] * vmf.pdf(points),
            atol=1e-10,
        )

    def test_pdf_4d(self):
        wad = WatsonDistribution(array([0.0, 0.0, 0.0, 1.0]), -10)
        vmf = VonMisesFisherDistribution(array([0.0, 1.0, 0.0, 0.0]), 1)
        w = array([0.3, 0.7])
        smix = HypersphericalMixture([wad, vmf], w)

        a, b, c, d = meshgrid(
            arange(-1, 4), arange(-1, 4), arange(-1, 4), arange(-1, 4)
        )
        points = array([a.ravel(), b.ravel(), c.ravel(), d.ravel()]).T
        points = points / sqrt(sum(points**2, axis=1, keepdims=True))

        assert_allclose(
            smix.pdf(points),
            w[0] * wad.pdf(points) + w[1] * vmf.pdf(points),
            atol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
