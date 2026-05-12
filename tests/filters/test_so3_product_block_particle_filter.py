import unittest

from pyrecest.filters import (
    BlockParticleFilter,
    PartitionedSO3ProductParticleFilter,
    SO3ProductBlockParticleFilter,
    SO3ProductParticleFilter,
)


class SO3ProductBlockParticleFilterTest(unittest.TestCase):
    def test_new_name_inherits_generic_block_filter(self):
        filt = SO3ProductBlockParticleFilter(
            n_particles=3,
            num_rotations=2,
            partition="singleton",
        )

        self.assertIsInstance(filt, BlockParticleFilter)
        self.assertIsInstance(filt, SO3ProductParticleFilter)
        self.assertEqual(filt.partition, ((0,), (1,)))

    def test_old_partitioned_name_is_backward_compatible(self):
        self.assertTrue(
            issubclass(
                PartitionedSO3ProductParticleFilter,
                SO3ProductBlockParticleFilter,
            )
        )
        filt = PartitionedSO3ProductParticleFilter(
            n_particles=2,
            num_rotations=2,
            partition="global",
        )
        self.assertIsInstance(filt, SO3ProductBlockParticleFilter)
        self.assertEqual(filt.partition, ((0, 1),))


if __name__ == "__main__":
    unittest.main()
