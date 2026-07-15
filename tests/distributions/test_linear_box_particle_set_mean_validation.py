import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions.nonperiodic.linear_box_particle_distribution import (
    LinearBoxParticleDistribution,
)


class LinearBoxParticleSetMeanValidationTest(unittest.TestCase):
    def test_set_mean_rejects_broadcastable_wrong_shapes(self):
        dist = LinearBoxParticleDistribution(array([[0.0, 0.0]]), array([[2.0, 4.0]]))

        for new_mean in (array(10.0), array([10.0]), array([[10.0, 11.0]])):
            with self.subTest(shape=new_mean.shape):
                with self.assertRaisesRegex(ValueError, "new_mean must have shape"):
                    dist.set_mean(new_mean)

    def test_set_mean_reaches_target_without_changing_box_widths(self):
        dist = LinearBoxParticleDistribution(
            array([[0.0, 0.0], [4.0, 2.0]]),
            array([[2.0, 4.0], [8.0, 6.0]]),
            array([0.25, 0.75]),
        )
        original_mean = dist.mean()
        original_widths = dist.widths()

        shifted = dist.set_mean(array([10.0, -3.0]))

        npt.assert_allclose(shifted.mean(), array([10.0, -3.0]))
        npt.assert_allclose(shifted.widths(), original_widths)
        npt.assert_allclose(dist.mean(), original_mean)

    def test_set_mean_accepts_scalar_for_one_dimension(self):
        dist = LinearBoxParticleDistribution(array([[0.0]]), array([[2.0]]))

        shifted = dist.set_mean(array(4.0))

        npt.assert_allclose(shifted.mean(), array([4.0]))


if __name__ == "__main__":
    unittest.main()
