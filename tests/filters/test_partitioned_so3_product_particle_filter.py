import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, linalg, ones, pi, sin, stack
from pyrecest.filters import PartitionedSO3ProductParticleFilter, SO3ProductParticleFilter

ATOL = 1e-6


def z_quaternion(angle):
    return array([0.0, 0.0, sin(angle / 2.0), cos(angle / 2.0)])


def x_quaternion(angle):
    return array([sin(angle / 2.0), 0.0, 0.0, cos(angle / 2.0)])


class PartitionedSO3ProductParticleFilterTest(unittest.TestCase):
    def test_initializes_partitioned_filter(self):
        filt = PartitionedSO3ProductParticleFilter(
            n_particles=5,
            num_rotations=3,
            partition=((0, 2), (1,)),
        )

        self.assertIsInstance(filt, SO3ProductParticleFilter)
        self.assertEqual(filt.partition, ((0, 2), (1,)))
        self.assertEqual(filt.block_weights.shape, (2, 5))
        npt.assert_allclose(filt.block_weights, ones((2, 5)) / 5)
        npt.assert_allclose(filt.weights, ones(5) / 5)
        npt.assert_allclose(filt.block_effective_sample_size(), array([5.0, 5.0]))
        npt.assert_allclose(filt.effective_sample_size(), 5.0)

    def test_named_partitions_and_validation(self):
        self.assertEqual(
            PartitionedSO3ProductParticleFilter._validate_partition("global", 3),
            ((0, 1, 2),),
        )
        self.assertEqual(
            PartitionedSO3ProductParticleFilter._validate_partition("singleton", 3),
            ((0,), (1,), (2,)),
        )
        with self.assertRaises(ValueError):
            PartitionedSO3ProductParticleFilter(2, 3, partition=((0, 1), (1, 2)))
        with self.assertRaises(ValueError):
            PartitionedSO3ProductParticleFilter(2, 3, partition=((0,), (2,)))

    def test_geodesic_update_uses_independent_block_weights(self):
        filt = PartitionedSO3ProductParticleFilter(
            n_particles=2,
            num_rotations=2,
            partition="singleton",
        )
        filt.set_particles(
            stack(
                [
                    array([[0.0, 0.0, 0.0, 1.0], z_quaternion(0.0)]),
                    array([[0.0, 0.0, 0.0, 1.0], z_quaternion(pi / 2.0)]),
                ],
                axis=0,
            )
        )

        ess = filt.update_with_geodesic_likelihood(
            array([[0.0, 0.0, 0.0, 1.0], z_quaternion(pi / 2.0)]),
            noise_std=0.2,
            resample=False,
        )

        self.assertGreater(float(filt.block_weights[0, 0]), 0.49)
        self.assertGreater(float(filt.block_weights[1, 1]), 0.99)
        self.assertLess(float(ess[1]), 2.0)
        estimate = filt.mean()
        npt.assert_allclose(estimate[0], array([0.0, 0.0, 0.0, 1.0]), atol=ATOL)
        npt.assert_allclose(estimate[1], z_quaternion(pi / 2.0), atol=ATOL)

    def test_resample_blocks_assembles_hybrid_particles(self):
        filt = PartitionedSO3ProductParticleFilter(
            n_particles=2,
            num_rotations=2,
            partition="singleton",
        )
        filt.set_particles(
            stack(
                [
                    array([[0.0, 0.0, 0.0, 1.0], z_quaternion(0.0)]),
                    array([x_quaternion(pi / 2.0), z_quaternion(pi / 2.0)]),
                ],
                axis=0,
            ),
            block_weights=array([[1.0, 0.0], [0.0, 1.0]]),
        )

        filt.resample_blocks_systematic()

        expected = array([[0.0, 0.0, 0.0, 1.0], z_quaternion(pi / 2.0)])
        npt.assert_allclose(filt.particles[0], expected, atol=ATOL)
        npt.assert_allclose(filt.particles[1], expected, atol=ATOL)
        npt.assert_allclose(linalg.norm(filt.particles, axis=-1), ones((2, 2)))
        npt.assert_allclose(filt.block_weights, ones((2, 2)) / 2)

    def test_component_likelihood_update(self):
        filt = PartitionedSO3ProductParticleFilter(
            n_particles=2,
            num_rotations=2,
            partition=((0, 1),),
        )

        filt.update_with_component_likelihoods(
            array([[1.0, 0.5], [0.25, 0.25]]),
            resample=False,
        )

        npt.assert_allclose(filt.block_weights, array([[8.0 / 9.0, 1.0 / 9.0]]))
        with self.assertRaises(ValueError):
            filt.update_with_component_likelihoods(array([[1.0, -1.0], [1.0, 1.0]]))


if __name__ == "__main__":
    unittest.main()
