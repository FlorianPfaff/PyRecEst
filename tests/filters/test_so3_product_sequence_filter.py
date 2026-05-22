import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.backend import array, cos, ones, sin, stack, to_numpy
from pyrecest.diagnostics import ParticleFilterResult
from pyrecest.filters import run_so3_product_sequence_filter


def z_quaternion(angle):
    return array([0.0, 0.0, sin(angle / 2.0), cos(angle / 2.0)])


def x_quaternion(angle):
    return array([sin(angle / 2.0), 0.0, 0.0, cos(angle / 2.0)])


class SO3ProductSequenceFilterTest(unittest.TestCase):
    def test_runs_global_filter_and_returns_diagnostics(self):
        measurements = stack(
            [
                stack([z_quaternion(0.0), x_quaternion(0.0)], axis=0),
                stack([z_quaternion(0.1), x_quaternion(0.0)], axis=0),
                stack([z_quaternion(0.2), x_quaternion(0.1)], axis=0),
            ],
            axis=0,
        )
        initial_particles = stack([measurements[0] for _ in range(8)], axis=0)
        mask = array([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        confidence = array([[1.0, 1.0], [0.75, 0.0], [1.0, 0.5]])

        result = run_so3_product_sequence_filter(
            measurements,
            mask,
            noise_std=0.25,
            num_particles=8,
            confidence=confidence,
            initial_particles=initial_particles,
            resample_threshold=0.0,
        )

        self.assertIsInstance(result, ParticleFilterResult)
        self.assertEqual(result.estimates.shape, measurements.shape)
        self.assertEqual(result.effective_sample_size.shape, (3,))
        self.assertEqual(result.resampled.shape, (3,))
        self.assertEqual(result.particle_spread.shape, (3,))
        self.assertIsNone(result.block_effective_sample_size)
        npt.assert_allclose(to_numpy(result.resampled), np.array([False, False, False]))

    def test_runs_singleton_block_filter_and_records_block_ess(self):
        measurements = stack(
            [
                stack([z_quaternion(0.0), x_quaternion(0.0)], axis=0),
                stack([z_quaternion(0.1), x_quaternion(0.2)], axis=0),
            ],
            axis=0,
        )
        initial_particles = stack([measurements[0] for _ in range(6)], axis=0)

        result = run_so3_product_sequence_filter(
            measurements,
            ones((2, 2)),
            noise_std=0.3,
            num_particles=6,
            partition="singleton",
            initial_particles=initial_particles,
            resample_threshold=0.0,
        )

        self.assertEqual(result.block_effective_sample_size.shape, (2, 2))
        self.assertEqual(result.metadata["partition"], ((0,), (1,)))

    def test_transition_callback_predicts_particles(self):
        measurements = stack(
            [
                stack([z_quaternion(0.0)], axis=0),
                stack([z_quaternion(0.1)], axis=0),
            ],
            axis=0,
        )
        initial_particles = stack([measurements[0] for _ in range(4)], axis=0)

        def transition(particles, time_index, rng):
            del time_index, rng
            return particles

        result = run_so3_product_sequence_filter(
            measurements,
            noise_std=0.3,
            transition_callback=transition,
            num_particles=4,
            initial_particles=initial_particles,
            resample_threshold=0.0,
        )

        self.assertEqual(result.estimates.shape, (2, 1, 4))


if __name__ == "__main__":
    unittest.main()
