"""Regression tests for atomic block-particle resampling validation."""

import copy
import unittest
from types import SimpleNamespace

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, ones
from pyrecest.filters import BlockParticleFilter


class _DummyBlockParticleFilter(BlockParticleFilter):
    def __init__(self, particles, partition=None, block_weights=None):
        particles = array(particles, dtype=float)
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


class BlockParticleResamplingAtomicityTest(unittest.TestCase):
    def test_all_block_indices_are_validated_before_resampling(self):
        filt = _DummyBlockParticleFilter(
            array([[0.0, 10.0], [1.0, 11.0]]),
            partition="singleton",
            block_weights=array([[1.0, 0.0], [0.0, 1.0]]),
        )
        original_particles = copy.deepcopy(filt.particles)
        original_weights = copy.deepcopy(filt.weights)
        original_block_weights = copy.deepcopy(filt.block_weights)

        with self.assertRaisesRegex(ValueError, "out of range"):
            filt.resample_blocks_systematic([0, filt.n_blocks])

        npt.assert_allclose(filt.particles, original_particles)
        npt.assert_allclose(filt.weights, original_weights)
        npt.assert_allclose(filt.block_weights, original_block_weights)


if __name__ == "__main__":
    unittest.main()
