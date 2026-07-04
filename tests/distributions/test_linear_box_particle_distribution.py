import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, ones, to_numpy
from pyrecest.distributions.nonperiodic.linear_box_particle_distribution import (
    LinearBoxParticleDistribution,
)


class LinearBoxParticleDistributionTest(unittest.TestCase):
    def test_mean_and_covariance_include_uniform_box_variance(self):
        dist = LinearBoxParticleDistribution(array([[0.0, 0.0]]), array([[2.0, 4.0]]))

        npt.assert_allclose(dist.mean(), array([1.0, 2.0]))
        npt.assert_allclose(
            dist.covariance(), array([[1.0 / 3.0, 0.0], [0.0, 4.0 / 3.0]])
        )

    def test_pdf_of_single_one_dimensional_box(self):
        dist = LinearBoxParticleDistribution(array([[0.0]]), array([[2.0]]))

        npt.assert_allclose(
            dist.pdf(array([-1.0, 0.5, 1.5, 3.0])), array([0.0, 0.5, 0.5, 0.0])
        )

    def test_sample_rejects_invalid_count(self):
        dist = LinearBoxParticleDistribution(array([[0.0]]), array([[2.0]]))

        for n in (0, -1, 1.5, True):
            with self.subTest(n=n):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    dist.sample(n)

    def test_integrate_query_box(self):
        dist = LinearBoxParticleDistribution(array([[0.0]]), array([[4.0]]), ones(1))

        npt.assert_allclose(dist.integrate(array([1.0]), array([3.0])), 0.5)

    def test_constructor_rejects_zero_volume_boxes(self):
        invalid_boxes = (
            (array([[0.0]]), array([[0.0]])),
            (array([[0.0, 0.0]]), array([[1.0, 0.0]])),
        )

        for lower, upper in invalid_boxes:
            with self.subTest(lower=lower, upper=upper):
                with self.assertRaisesRegex(ValueError, "upper > lower"):
                    LinearBoxParticleDistribution(lower, upper)

    def test_constructor_rejects_nonfinite_weights(self):
        invalid_weights = (
            ("nan", array([np.nan, 1.0])),
            ("inf", array([np.inf, 1.0])),
        )

        for name, weights in invalid_weights:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "finite"):
                    LinearBoxParticleDistribution(
                        array([[0.0], [1.0]]), array([[1.0], [2.0]]), weights
                    )

    def test_reweigh_rejects_invalid_weight_updates(self):
        invalid_updates = (
            ("nan", lambda _centers: array([1.0, np.nan]), "finite"),
            ("inf", lambda _centers: array([1.0, np.inf]), "finite"),
            ("negative", lambda _centers: array([1.0, -0.5]), "nonnegative"),
            (
                "zero-mass",
                lambda _centers: array([0.0, 0.0]),
                "positive finite total mass",
            ),
        )

        for name, update, message in invalid_updates:
            with self.subTest(name=name):
                dist = LinearBoxParticleDistribution(
                    array([[0.0], [1.0]]), array([[1.0], [2.0]])
                )

                with self.assertRaisesRegex(ValueError, message):
                    dist.reweigh(update)

                npt.assert_allclose(to_numpy(dist.w), [0.5, 0.5])

    def test_from_distribution_rejects_invalid_particle_count_aliases(self):
        for n_particles in (0, -1, 1.5, True):
            with self.subTest(n_particles=n_particles):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    LinearBoxParticleDistribution.from_distribution(
                        object(), n_particles=n_particles
                    )


if __name__ == "__main__":
    unittest.main()
