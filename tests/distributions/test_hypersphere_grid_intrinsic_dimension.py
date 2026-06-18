import unittest

import numpy.testing as npt
import pyrecest
from pyrecest.backend import array, ones, pi
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_grid_distribution import (
    HyperhemisphericalGridDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_uniform_distribution import (
    HyperhemisphericalUniformDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperspherical_grid_distribution import (
    HypersphericalGridDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperspherical_uniform_distribution import (
    HypersphericalUniformDistribution,
)
from pyrecest.distributions.hypersphere_subset.spherical_grid_distribution import (
    SphericalGridDistribution,
)


class HypersphereGridIntrinsicDimensionTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_hyperspherical_s2_grid_uses_intrinsic_dimension(self):
        dist = HypersphericalUniformDistribution(2)
        hgd = HypersphericalGridDistribution.from_distribution(dist, 1000, "leopardi")

        self.assertEqual(hgd.dim, 2)
        self.assertEqual(hgd.input_dim, 3)
        self.assertEqual(hgd.get_grid().shape[1], hgd.input_dim)
        npt.assert_allclose(hgd.get_manifold_size(), 4 * pi, rtol=0, atol=1e-12)
        npt.assert_allclose(hgd.integrate(), 1.0, rtol=0, atol=1e-12)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_spherical_grid_uses_intrinsic_dimension(self):
        dist = HypersphericalUniformDistribution(2)
        sgd = SphericalGridDistribution.from_distribution(dist, 1000, "leopardi")

        self.assertEqual(sgd.dim, 2)
        self.assertEqual(sgd.input_dim, 3)
        self.assertEqual(sgd.get_grid().shape[1], sgd.input_dim)
        npt.assert_allclose(sgd.get_manifold_size(), 4 * pi, rtol=0, atol=1e-12)
        npt.assert_allclose(sgd.integrate(), 1.0, rtol=0, atol=1e-12)
        npt.assert_allclose(sgd.pdf(sgd.get_grid()[:3]), sgd.grid_values[:3])

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_spherical_grid_rejects_unsupported_harmonic_interpolation(self):
        dist = SphericalGridDistribution.from_function(
            lambda xs: ones(xs.shape[0]),
            42,
        )

        with self.assertRaisesRegex(NotImplementedError, "harmonics"):
            dist.pdf(array([0.0, 0.0, 1.0]), use_harmonics=True)

        with self.assertRaisesRegex(NotImplementedError, "harmonics"):
            dist.plot_interpolated(use_harmonics=True)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_spherical_grid_rejects_wrong_intrinsic_dimension(self):
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            SphericalGridDistribution.from_distribution(
                HypersphericalUniformDistribution(3),
                42,
                "leopardi",
            )

        with self.assertRaisesRegex(ValueError, "dimensions other than 2"):
            SphericalGridDistribution.from_function(
                lambda xs: ones(xs.shape[0]),
                42,
                dim=3,
            )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_hyperhemispherical_s2_grid_uses_intrinsic_dimension(self):
        dist = HyperhemisphericalUniformDistribution(2)
        hhgd = HyperhemisphericalGridDistribution.from_distribution(
            dist, 42, "leopardi_symm"
        )

        self.assertEqual(hhgd.dim, 2)
        self.assertEqual(hhgd.input_dim, 3)
        self.assertEqual(hhgd.get_grid().shape[1], hhgd.input_dim)
        npt.assert_allclose(hhgd.get_manifold_size(), 2 * pi, rtol=0, atol=1e-12)
        npt.assert_allclose(hhgd.integrate(), 1.0, rtol=0, atol=1e-12)
        point, _ = hhgd.get_closest_point(hhgd.get_grid()[0])
        npt.assert_allclose(point, hhgd.get_grid()[0], rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
