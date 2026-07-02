"""Regression tests for transition samplers without sample-count support."""

import unittest

import numpy as np
from pyrecest.backend import array
from pyrecest.models import (
    DensityTransitionModel,
    SampleableTransitionModel,
    sample_next_state,
)


class UnsupportedTransitionSampleCountTest(unittest.TestCase):
    def test_sampleable_transition_rejects_count_when_sampler_has_no_count_argument(self):
        calls = []

        def sample_next(state):
            calls.append(state)
            return state

        model = SampleableTransitionModel(sample_next)
        state = array([0.0])

        with self.assertRaisesRegex(TypeError, "sample count is not supported"):
            sample_next_state(model, state, n=2)

        self.assertEqual(calls, [])
        self.assertIsNotNone(sample_next_state(model, state, n=1))
        self.assertEqual(len(calls), 1)

    def test_density_transition_rejects_count_when_sampler_has_no_count_argument(self):
        calls = []

        def transition_density(state_next, state_previous):
            return 1.0 if state_next is state_previous else 0.0

        def sample_next(state):
            calls.append(state)
            return state

        model = DensityTransitionModel(transition_density, sample_next=sample_next)
        state = array([0.0])

        with self.assertRaisesRegex(TypeError, "sample count is not supported"):
            model.sample_next(state, n=2)

        self.assertEqual(calls, [])
        self.assertIsNotNone(model.sample_next(state, n=1))
        self.assertEqual(len(calls), 1)

    def test_sampler_without_count_still_rejects_invalid_counts_cleanly(self):
        model = SampleableTransitionModel(lambda state: state)

        with self.assertRaisesRegex(ValueError, "n must be a nonnegative integer"):
            sample_next_state(model, array([0.0]), n=np.array([2]))


if __name__ == "__main__":
    unittest.main()
