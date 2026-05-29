import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.backend import array, cos, ones, sin, stack, to_numpy
from pyrecest.diagnostics import ParticleFilterResult
from pyrecest.filters import (
    SO3ProductSequenceFilterRunner,
    run_so3_product_sequence_filter,
)


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

    def test_runner_accepts_component_noise_at_run_time(self):
        measurements = stack(
            [
                stack([z_quaternion(0.0)], axis=0),
                stack([z_quaternion(0.1)], axis=0),
            ],
            axis=0,
        )
        initial_particles = stack([measurements[0] for _ in range(4)], axis=0)
        runner = SO3ProductSequenceFilterRunner(
            num_particles=4,
            resample_threshold=0.0,
        )

        result = runner.run(
            measurements,
            component_noise_std=0.3,
            initial_particles=initial_particles,
        )

        self.assertEqual(result.estimates.shape, (2, 1, 4))

    def test_confidence_can_drive_noise_mapping(self):
        measurements = stack(
            [
                stack([z_quaternion(0.0), x_quaternion(0.0)], axis=0),
                stack([z_quaternion(0.2), x_quaternion(0.1)], axis=0),
            ],
            axis=0,
        )
        confidence = array([[1.0, 0.2], [0.8, 0.1]])
        initial_particles = stack([measurements[0] for _ in range(5)], axis=0)

        result = run_so3_product_sequence_filter(
            measurements,
            noise_std=0.1,
            max_noise_std=0.5,
            confidence=confidence,
            num_particles=5,
            initial_particles=initial_particles,
            resample_threshold=0.0,
        )

        self.assertEqual(result.estimates.shape, measurements.shape)
        self.assertEqual(result.metadata["particle_spread_unit"], "rad")

    def test_validates_noise_and_proposal_parameters(self):
        measurements = stack([stack([z_quaternion(0.0)], axis=0)], axis=0)

        with self.assertRaises(ValueError):
            run_so3_product_sequence_filter(measurements, noise_std=-0.1)
        with self.assertRaises(ValueError):
            run_so3_product_sequence_filter(
                measurements, noise_std=0.1, num_particles=1.5
            )
        with self.assertRaises(ValueError):
            run_so3_product_sequence_filter(
                measurements, noise_std=0.1, max_noise_std=-0.5
            )
        with self.assertRaises(ValueError):
            run_so3_product_sequence_filter(
                measurements,
                noise_std=0.1,
                component_noise_std=0.1,
                max_noise_std=0.5,
            )
        with self.assertRaises(ValueError):
            run_so3_product_sequence_filter(
                measurements, noise_std=0.5, max_noise_std=0.1
            )
        with self.assertRaises(ValueError):
            run_so3_product_sequence_filter(
                measurements, noise_std=0.1, initial_noise_std=-0.01
            )
        with self.assertRaises(ValueError):
            run_so3_product_sequence_filter(
                measurements, noise_std=0.1, proposal_gain=-0.2
            )
        with self.assertRaises(ValueError):
            run_so3_product_sequence_filter(
                measurements, noise_std=0.1, resample_threshold=-0.5
            )
        with self.assertRaises(ValueError):
            run_so3_product_sequence_filter(
                measurements, noise_std=0.1, confidence_exponent=0.0
            )
        with self.assertRaises(ValueError):
            run_so3_product_sequence_filter(
                measurements, noise_std=0.1, outlier_prob=1.5
            )
        with self.assertRaises(ValueError):
            run_so3_product_sequence_filter(
                measurements, noise_std=0.1, transition_callback="not-callable"
            )

    def test_runner_validates_configuration_at_construction(self):
        invalid_kwargs = [
            {"noise_std": 0.1, "num_particles": 0},
            {"noise_std": 0.1, "num_particles": 1.5},
            {"noise_std": 0.1, "num_particles": True},
            {"noise_std": 0.1, "resample_threshold": -0.5},
            {"noise_std": 0.1, "proposal_gain": -0.2},
            {"noise_std": 0.1, "initial_noise_std": -0.01},
            {"noise_std": 0.1, "max_noise_std": -0.5},
            {"noise_std": 0.5, "max_noise_std": 0.1},
            {"noise_std": 0.1, "confidence_exponent": 0.0},
            {"noise_std": 0.1, "outlier_prob": 1.5},
            {"noise_std": 0.1, "transition_callback": "not-callable"},
            {"max_noise_std": 0.5},
        ]

        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SO3ProductSequenceFilterRunner(**kwargs)

    def test_runner_rejects_conflicting_runtime_noise_modes(self):
        measurements = stack([stack([z_quaternion(0.0)], axis=0)], axis=0)
        runner = SO3ProductSequenceFilterRunner(noise_std=0.1, max_noise_std=0.5)

        with self.assertRaises(ValueError):
            runner.run(measurements, component_noise_std=0.1)


if __name__ == "__main__":
    unittest.main()
