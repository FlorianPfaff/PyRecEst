"""Regression tests for additive-noise model sample-count validation."""

import unittest

import numpy as np
from pyrecest.backend import array, zeros
from pyrecest.models import AdditiveNoiseMeasurementModel, AdditiveNoiseTransitionModel


class RecordingNoise:
    def __init__(self):
        self.sample_counts = []

    def sample(self, n):
        self.sample_counts.append(n)
        return zeros((n, 1))


class AdditiveNoiseSampleCountValidationTest(unittest.TestCase):
    @staticmethod
    def _model_cases(noise):
        return (
            (
                AdditiveNoiseTransitionModel(
                    lambda state: state,
                    noise_distribution=noise,
                ),
                "sample_next",
            ),
            (
                AdditiveNoiseMeasurementModel(
                    lambda state: state,
                    noise_distribution=noise,
                ),
                "sample_measurement",
            ),
        )

    def test_invalid_counts_are_rejected_before_sampling(self):
        invalid_counts = (True, -1, 1.5, np.array([1]), "1")

        for invalid_count in invalid_counts:
            for model_index in range(2):
                noise = RecordingNoise()
                model, method_name = self._model_cases(noise)[model_index]
                with self.subTest(
                    invalid_count=invalid_count,
                    method_name=method_name,
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        "n must be a nonnegative integer",
                    ):
                        getattr(model, method_name)(array([1.0]), n=invalid_count)
                    self.assertEqual(noise.sample_counts, [])

    def test_integer_like_scalar_count_is_normalized_before_sampling(self):
        for model_index in range(2):
            noise = RecordingNoise()
            model, method_name = self._model_cases(noise)[model_index]
            with self.subTest(method_name=method_name):
                result = getattr(model, method_name)(array([1.0]), n=np.int64(2))

                self.assertEqual(noise.sample_counts, [2])
                self.assertIs(type(noise.sample_counts[0]), int)
                self.assertEqual(result.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
