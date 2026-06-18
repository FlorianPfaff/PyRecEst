import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, concatenate, diag, linalg, sum, tile
from pyrecest.distributions import (
    GaussianDistribution,
    HyperhemisphericalUniformDistribution,
)
from pyrecest.distributions.abstract_se3_distribution import AbstractSE3Distribution
from pyrecest.distributions.se3_cart_prod_stacked_distribution import (
    SE3CartProdStackedDistribution,
)
from pyrecest.distributions.se3_dirac_distribution import SE3DiracDistribution


class SE3DiracDistributionTest(unittest.TestCase):
    def test_plot_trajectory_rejects_mismatched_sample_counts(self):
        periodic_states = array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        lin_states = array([[0.0], [0.0], [0.0]])

        with self.assertRaisesRegex(ValueError, "same number of samples"):
            AbstractSE3Distribution.plot_trajectory(periodic_states, lin_states)

    def test_constructor(self):
        dSph = array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [2.0, 4.0, 0.0, 0.5, 1.0, 1.0],
                [5.0, 10.0, 20, 30, 40, 50],
                [2.0, 31.0, 42, 3, 9.9, 5],
            ]
        ).T
        dSph = dSph / linalg.norm(dSph, None, -1).reshape(-1, 1)
        dLin = tile(array([-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]), (3, 1)).T
        w = array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        w = w / sum(w)
        SE3DiracDistribution(concatenate((dSph, dLin), axis=-1), w)

    def test_constructor_accepts_list_inputs(self):
        particles = [
            [1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 0.0, 0.0, 4.0, 5.0, 6.0],
        ]
        weights = [0.25, 0.75]

        dist = SE3DiracDistribution(particles, weights)

        npt.assert_allclose(dist.d, array(particles))
        npt.assert_allclose(dist.w, array(weights))
        self.assertEqual(dist.d.shape, (2, 7))

    def test_constructor_rejects_non_unit_hypersphere_particles(self):
        particles = array([[2.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]])

        with self.assertRaisesRegex(ValueError, "normalized"):
            SE3DiracDistribution(particles)

    def test_from_distribution(self):
        cpsd = SE3CartProdStackedDistribution(
            [
                HyperhemisphericalUniformDistribution(3),
                GaussianDistribution(
                    array([1.0, 2.0, 3.0]).T, diag(array([3.0, 2.0, 1.0]))
                ),
            ]
        )
        ddist = SE3DiracDistribution.from_distribution(cpsd, np.int64(3))
        self.assertEqual(ddist.d.shape, (3, 7))
        self.assertEqual(ddist.w.shape, (3,))

        for n_particles in (True, 1.5, 0, -1):
            with self.subTest(n_particles=n_particles):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    SE3DiracDistribution.from_distribution(cpsd, n_particles)

        with self.assertRaises(TypeError):
            SE3DiracDistribution.from_distribution("not a distribution", 3)


if __name__ == "__main__":
    unittest.main()
