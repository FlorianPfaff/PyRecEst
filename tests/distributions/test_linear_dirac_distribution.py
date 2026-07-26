import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from parameterized import parameterized

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, diag, eye, random
from pyrecest.distributions import GaussianDistribution
from pyrecest.distributions.nonperiodic.linear_dirac_distribution import (
    LinearDiracDistribution,
)
from scipy.stats import wishart

from .test_abstract_dirac_distribution import TestAbstractDiracDistribution


class LinearDiracDistributionTest(TestAbstractDiracDistribution):
    def test_constructor_accepts_list_inputs(self):
        one_dimensional = LinearDiracDistribution([1.0, 2.0, 3.0])
        multidimensional = LinearDiracDistribution(
            [[1.0, 2.0], [3.0, 4.0]], [0.25, 0.75]
        )

        self.assertEqual(one_dimensional.dim, 1)
        npt.assert_allclose(one_dimensional.d, array([1.0, 2.0, 3.0]))
        self.assertEqual(multidimensional.dim, 2)
        npt.assert_allclose(multidimensional.d, array([[1.0, 2.0], [3.0, 4.0]]))
        npt.assert_allclose(multidimensional.w, array([0.25, 0.75]))

    def test_constructor_accepts_scalar_location(self):
        scalar_dist = LinearDiracDistribution(2.5)

        self.assertEqual(scalar_dist.dim, 1)
        npt.assert_allclose(scalar_dist.d, array([2.5]))
        npt.assert_allclose(scalar_dist.w, array([1.0]))
        npt.assert_allclose(scalar_dist.covariance(), array([[0.0]]))

    def test_weighted_samples_to_mean_and_cov_accepts_scalar_sample(self):
        mean, covariance = LinearDiracDistribution.weighted_samples_to_mean_and_cov(2.5)

        npt.assert_allclose(mean, array([2.5]))
        npt.assert_allclose(covariance, array([[0.0]]))

    def test_set_mean_rejects_invalid_multidimensional_shapes(self):
        dist = LinearDiracDistribution([[0.0, 0.0], [2.0, 2.0]], [0.5, 0.5])
        expected_samples = array([[0.0, 0.0], [2.0, 2.0]])

        for name, new_mean in (
            ("scalar", 1.0),
            ("too-short", array([1.0])),
            ("too-long", array([1.0, 2.0, 3.0])),
            ("column-vector", array([[1.0], [2.0]])),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    ValueError, r"new_mean must have shape \(2,\)"
                ):
                    dist.set_mean(new_mean)
                npt.assert_allclose(dist.d, expected_samples)

    def test_set_mean_accepts_scalar_for_one_dimensional_distribution(self):
        dist = LinearDiracDistribution([0.0, 2.0], [0.25, 0.75])

        dist.set_mean(10.0)

        npt.assert_allclose(dist.mean(), array([10.0]))
        npt.assert_allclose(dist.d, array([8.5, 10.5]))

    def test_from_distribution(self):
        random.seed(0)
        C = wishart.rvs(3, eye(3))
        hwn = GaussianDistribution(array([1.0, 2.0, 3.0]), array(C))
        hwd = LinearDiracDistribution.from_distribution(hwn, 200000)
        npt.assert_allclose(hwd.mean(), hwn.mean(), atol=0.015)
        npt.assert_allclose(hwd.covariance(), hwn.covariance(), atol=0.1)

    def test_mean_and_cov(self):
        random.seed(0)
        gd = GaussianDistribution(array([1.0, 2.0]), array([[2.0, -0.3], [-0.3, 1.0]]))
        ddist = LinearDiracDistribution(gd.sample(15000))
        self.assertTrue(allclose(ddist.mean(), gd.mean(), atol=0.05))
        self.assertTrue(allclose(ddist.covariance(), gd.covariance(), atol=0.05))

    def test_weighted_samples_to_mean_and_cov_rejects_invalid_weights(self):
        samples = array([[0.0], [2.0]])
        invalid_weights = (
            ("nan", array([1.0, np.nan]), "finite"),
            ("inf", array([1.0, np.inf]), "finite"),
            ("negative", array([1.5, -0.5]), "nonnegative"),
            ("zero-mass", array([0.0, 0.0]), "positive finite total mass"),
        )

        for name, weights, message in invalid_weights:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    LinearDiracDistribution.weighted_samples_to_mean_and_cov(
                        samples, weights
                    )

    @parameterized.expand(
        [
            (
                "1D Plot",
                GaussianDistribution(
                    array([1.0]), array([[1.0]])  # 1D mean
                ),  # 1D covariance
                1,  # Dimension
            ),
            (
                "2D Plot",
                GaussianDistribution(
                    array([1.0, 2.0]), array([[2.0, -0.3], [-0.3, 1.0]])  # 2D mean
                ),  # 2D covariance
                2,  # Dimension
            ),
            (
                "3D Plot",
                GaussianDistribution(
                    array([1.0, 2.0, 3.0]), diag(array([2.0, 1.0, 0.5]))  # 3D mean
                ),  # 3D covariance (diagonal matrix)
                3,  # Dimension
            ),
        ]
    )
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_plot(self, name, dist, dim):
        self._test_plot_helper(name, dist, dim, LinearDiracDistribution)


if __name__ == "__main__":
    unittest.main()
