import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, linalg, mod, ones, pi, random, sqrt, sum
from pyrecest.distributions import VonMisesFisherDistribution
from pyrecest.distributions.hypersphere_subset.hyperspherical_dirac_distribution import (
    HypersphericalDiracDistribution,
)


class HypersphericalDiracDistributionTest(unittest.TestCase):
    def setUp(self):
        self.d = array(
            [
                [0.5, 3.0, 4.0, 6.0, 6.0],
                [2.0, 2.0, 5.0, 3.0, 0.0],
                [0.5, 0.2, 5.8, 4.3, 1.2],
            ]
        ).T
        self.d = self.d / linalg.norm(self.d, axis=1)[:, None]
        self.w = array([0.1, 0.1, 0.1, 0.1, 0.6])
        self.hdd = HypersphericalDiracDistribution(self.d, self.w)

    def test_instance_creation(self):
        self.assertIsInstance(self.hdd, HypersphericalDiracDistribution)

    def test_sampling(self):
        nSamples = 5
        s = self.hdd.sample(nSamples)
        self.assertEqual(s.shape, (nSamples, self.d.shape[-1]))
        npt.assert_array_almost_equal(s, mod(s, 2 * pi))
        npt.assert_array_almost_equal(linalg.norm(s, axis=-1), ones(nSamples))

    def test_apply_function(self):
        same = self.hdd.apply_function(lambda x: x)
        npt.assert_array_almost_equal(same.d, self.hdd.d, decimal=10)
        npt.assert_array_almost_equal(same.w, self.hdd.w, decimal=10)

        mirrored = self.hdd.apply_function(lambda x: -x)
        npt.assert_array_almost_equal(mirrored.d, -self.hdd.d, decimal=10)
        npt.assert_array_almost_equal(mirrored.w, self.hdd.w, decimal=10)

    def test_reweigh_identity(self):
        def f(x):
            return 2 * ones(x.shape[0])

        twdNew = self.hdd.reweigh(f)
        self.assertIsInstance(twdNew, HypersphericalDiracDistribution)
        npt.assert_array_almost_equal(twdNew.d, self.hdd.d)
        npt.assert_array_almost_equal(twdNew.w, self.hdd.w)

    def test_reweigh_coord_based(self):
        def f(x):
            return x[:, 1]

        twdNew = self.hdd.reweigh(f)
        self.assertIsInstance(twdNew, HypersphericalDiracDistribution)
        npt.assert_array_almost_equal(twdNew.d, self.hdd.d)
        self.assertAlmostEqual(sum(twdNew.w), 1, places=10)
        wNew = self.hdd.d[:, 1] * self.hdd.w
        npt.assert_array_almost_equal(twdNew.w, wNew / sum(wNew))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("jax", "pytorch"),
        reason="Not supported on this backend",
    )
    def test_from_distribution(self):
        random.seed(0)
        vmf = VonMisesFisherDistribution(array([1.0, 1.0, 1.0]) / sqrt(3), 1.0)
        dirac_dist = HypersphericalDiracDistribution.from_distribution(vmf, 100000)
        npt.assert_almost_equal(
            dirac_dist.mean_direction(), vmf.mean_direction(), decimal=2
        )

    def test_mean_axis_symmetric_two_point_distribution(self):
        # Two antipodal points on S²: ±e_x
        d = array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        )
        w = array([0.5, 0.5])

        dist = HypersphericalDiracDistribution(d, w)

        axis = dist.mean_axis()

        # 1) axis should be unit length
        assert allclose(linalg.norm(axis), 1.0, atol=1e-7)

        # 2) axis should be parallel to (1, 0, 0), i.e. |dot(axis, e_x)| ≈ 1
        v = array([1.0, 0.0, 0.0])
        dot = float(axis @ v)
        assert abs(dot) > 1.0 - 1e-6


if __name__ == "__main__":
    unittest.main()
