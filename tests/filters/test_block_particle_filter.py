import unittest
from types import SimpleNamespace

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, ones
from pyrecest.filters import BlockParticleFilter


class DummyBlockParticleFilter(BlockParticleFilter):
    def __init__(self, particles, partition=None, weights=None, block_weights=None):
        particles = array(particles, dtype=float)
        if weights is None:
            weights = ones(particles.shape[0]) / particles.shape[0]
        self.filter_state = SimpleNamespace(d=particles, w=weights)
        self._initialize_block_particle_filter(
            partition=partition,
            weights=weights,
            block_weights=block_weights,
        )

    @property
    def n_particles(self):
        return self.filter_state.d.shape[0]

    @property
    def particles(self):
        return self.filter_state.d

    @property
    def weights(self):
        return self.filter_state.w


class BlockParticleFilterTest(unittest.TestCase):
    def test_initializes_named_partitions(self):
        filt = DummyBlockParticleFilter(
            array([[0.0, 10.0, 20.0], [1.0, 11.0, 21.0]]),
            partition="singleton",
        )

        self.assertEqual(filt.partition, ((0,), (1,), (2,)))
        self.assertEqual(filt.n_blocks, 3)
        npt.assert_allclose(filt.block_weights, ones((3, 2)) / 2)
        npt.assert_allclose(filt.block_effective_sample_size(), array([2.0, 2.0, 2.0]))
        npt.assert_allclose(filt.effective_sample_size(), 2.0)

    def test_component_likelihoods_update_blocks_independently(self):
        filt = DummyBlockParticleFilter(
            array([[0.0, 10.0], [1.0, 11.0]]),
            partition="singleton",
        )

        filt.update_with_component_likelihoods(
            array([[1.0, 0.25], [0.25, 1.0]]),
            resample=False,
        )

        npt.assert_allclose(filt.block_weights, array([[0.8, 0.2], [0.2, 0.8]]))
        npt.assert_allclose(filt.weights, array([0.5, 0.5]))
        npt.assert_allclose(filt.component_weights(0), array([0.8, 0.2]))
        npt.assert_allclose(filt.component_weights(1), array([0.2, 0.8]))

    def test_resampling_assembles_hybrid_product_particles(self):
        filt = DummyBlockParticleFilter(
            array([[0.0, 10.0], [1.0, 11.0]]),
            partition="singleton",
            block_weights=array([[1.0, 0.0], [0.0, 1.0]]),
        )

        filt.resample_blocks_systematic()

        npt.assert_allclose(filt.particles, array([[0.0, 11.0], [0.0, 11.0]]))
        npt.assert_allclose(filt.block_weights, ones((2, 2)) / 2)
        npt.assert_allclose(filt.weights, ones(2) / 2)

    def test_global_log_likelihood_updates_all_blocks(self):
        filt = DummyBlockParticleFilter(
            array([[0.0, 10.0], [1.0, 11.0]]),
            partition="singleton",
        )

        filt.update_with_log_likelihood(array([0.0, -float("inf")]), resample=False)

        npt.assert_allclose(filt.block_weights, array([[1.0, 0.0], [1.0, 0.0]]))
        npt.assert_allclose(filt.weights, array([1.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
