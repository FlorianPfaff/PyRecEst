import unittest
from unittest.mock import patch

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

    def test_plot_point_draws_rotated_body_axes(self):
        class RecordingAxes:
            def __init__(self):
                self.quiver_calls = []

            def quiver(self, *args, **_kwargs):
                self.quiver_calls.append(args)
                return object()

            @staticmethod
            def get_xlim():
                return (-100.0, 100.0)

            @staticmethod
            def get_ylim():
                return (-100.0, 100.0)

            @staticmethod
            def get_zlim():
                return (-100.0, 100.0)

            @staticmethod
            def set_xlim(_limits):
                return None

            @staticmethod
            def set_ylim(_limits):
                return None

            @staticmethod
            def set_zlim(_limits):
                return None

        class RecordingFigure:
            def __init__(self, axes):
                self.axes = axes

            def add_subplot(self, *_args, **_kwargs):
                return self.axes

        axes = RecordingAxes()
        quarter_turn = np.sqrt(0.5)
        point = array([quarter_turn, 0.0, 0.0, quarter_turn, 10.0, 20.0, 30.0])

        with patch(
            "pyrecest.distributions.abstract_se3_distribution.plt.figure",
            return_value=RecordingFigure(axes),
        ):
            AbstractSE3Distribution.plot_point(point)

        directions = np.asarray([call[3:6] for call in axes.quiver_calls], dtype=float)
        npt.assert_allclose(
            directions,
            np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            atol=1e-12,
        )

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

    def test_linear_marginal_starts_after_full_quaternion_coordinates(self):
        particles = array(
            [
                [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0],
                [1.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0],
            ]
        )
        weights = array([0.25, 0.75])

        dist = SE3DiracDistribution(particles, weights)
        linear_marginal = dist.marginalize_periodic()

        self.assertEqual(linear_marginal.d.shape, (2, 3))
        npt.assert_allclose(linear_marginal.d, particles[:, 4:])
        npt.assert_allclose(dist.linear_mean(), array([3.25, 4.25, 5.25]))

    def test_constructor_rejects_unnormalized_hypersphere_particles(self):
        particles = array([[2.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]])

        with self.assertRaisesRegex(ValueError, "must be normalized"):
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
