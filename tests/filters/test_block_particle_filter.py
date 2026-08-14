import copy
import unittest
from types import SimpleNamespace

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, ones, to_numpy
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

    def test_set_particles_rejects_invalid_weights_atomically(self):
        replacement_particles = array([[2.0, 12.0], [3.0, 13.0]])
        invalid_cases = (
            ({"weights": array([1.0])}, "weights must match"),
            (
                {"block_weights": array([[1.0, 0.0]])},
                "block_weights must have shape",
            ),
        )

        for kwargs, error_message in invalid_cases:
            with self.subTest(kwargs=kwargs):
                filt = DummyBlockParticleFilter(
                    array([[0.0, 10.0], [1.0, 11.0]]),
                    partition="singleton",
                )
                original_particles = copy.deepcopy(filt.particles)
                original_weights = copy.deepcopy(filt.weights)
                original_block_weights = copy.deepcopy(filt.block_weights)

                with self.assertRaisesRegex(ValueError, error_message):
                    filt.set_particles(replacement_particles, **kwargs)

                npt.assert_allclose(filt.particles, original_particles)
                npt.assert_allclose(filt.weights, original_weights)
                npt.assert_allclose(filt.block_weights, original_block_weights)

    def test_component_indices_must_be_integral(self):
        filt = DummyBlockParticleFilter(
            array([[0.0, 10.0], [1.0, 11.0]]),
            partition="singleton",
        )

        for invalid_index in (True, 0.5, "0", [0]):
            with self.subTest(component_idx=invalid_index), self.assertRaises(
                ValueError
            ):
                filt.component_weights(invalid_index)

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

    def test_resample_block_indices_must_be_integral(self):
        filt = DummyBlockParticleFilter(
            array([[0.0, 10.0], [1.0, 11.0]]),
            partition="singleton",
            block_weights=array([[1.0, 0.0], [0.0, 1.0]]),
        )

        for invalid_index in (True, 0.5, "0", [0]):
            with self.subTest(block_index=invalid_index), self.assertRaises(ValueError):
                filt.resample_block_systematic(invalid_index)
            with self.subTest(block_indices=invalid_index), self.assertRaises(
                ValueError
            ):
                filt.resample_blocks_systematic([invalid_index])

    def test_normalizes_extreme_finite_weights_without_overflow(self):
        backend_dtype = np.asarray(to_numpy(array([1.0]))).dtype
        max_float = np.finfo(backend_dtype).max
        filt = DummyBlockParticleFilter(
            array([[0.0, 10.0], [1.0, 11.0]]),
            partition="singleton",
            weights=array([max_float, max_float / 2.0]),
        )

        expected = np.array([2.0 / 3.0, 1.0 / 3.0])
        npt.assert_allclose(to_numpy(filt.weights), expected)
        npt.assert_allclose(to_numpy(filt.block_weights), np.tile(expected, (2, 1)))

    def test_rejects_nonfinite_weights(self):
        for invalid_weight in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid_weight=invalid_weight):
                with self.assertRaisesRegex(ValueError, "finite"):
                    DummyBlockParticleFilter(
                        array([[0.0, 10.0], [1.0, 11.0]]),
                        weights=array([invalid_weight, 1.0]),
                    )

                with self.assertRaisesRegex(ValueError, "finite"):
                    DummyBlockParticleFilter(
                        array([[0.0, 10.0], [1.0, 11.0]]),
                        partition="singleton",
                        block_weights=array([[1.0, invalid_weight], [1.0, 1.0]]),
                    )

    def test_global_log_likelihood_updates_all_blocks(self):
        filt = DummyBlockParticleFilter(
            array([[0.0, 10.0], [1.0, 11.0]]),
            partition="singleton",
        )

        filt.update_with_log_likelihood(array([0.0, -float("inf")]), resample=False)

        npt.assert_allclose(filt.block_weights, array([[1.0, 0.0], [1.0, 0.0]]))
        npt.assert_allclose(filt.weights, array([1.0, 0.0]))

    def test_update_validates_ess_thresholds_before_mutating(self):
        initial_block_weights = array([[0.5, 0.5], [0.5, 0.5]])
        initial_weights = array([0.5, 0.5])
        invalid_thresholds = [
            -0.1,
            float("nan"),
            float("inf"),
            array([0.5, -0.1]),
            array([0.5, float("nan")]),
        ]

        for ess_threshold in invalid_thresholds:
            filt = DummyBlockParticleFilter(
                array([[0.0, 10.0], [1.0, 11.0]]),
                partition="singleton",
            )
            with self.subTest(ess_threshold=ess_threshold), self.assertRaises(
                ValueError
            ):
                filt.update_with_log_likelihood(
                    array([0.0, -1.0]),
                    ess_threshold=ess_threshold,
                    resample=False,
                )
            npt.assert_allclose(filt.block_weights, initial_block_weights)
            npt.assert_allclose(filt.weights, initial_weights)

    def test_update_accepts_scalar_and_per_block_ess_thresholds(self):
        filt = DummyBlockParticleFilter(
            array([[0.0, 10.0], [1.0, 11.0]]),
            partition="singleton",
        )

        filt.update_with_log_likelihood(
            array([0.0, -1.0]),
            ess_threshold=array([0.0, 0.0]),
            resample=True,
        )

        npt.assert_allclose(filt.block_weights, array([[0.73105858, 0.26894142]] * 2))


if __name__ == "__main__":
    unittest.main()
