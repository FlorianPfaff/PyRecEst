"""Regression tests for transition-sampler sample-count validation."""

import unittest

import numpy as np
from pyrecest.backend import array
from pyrecest.models import (
    DensityTransitionModel,
    SampleableTransitionModel,
    sample_next_state,
)


class TransitionSampleCountValidationTest(unittest.TestCase):
    def test_sampleable_transition_model_rejects_invalid_sample_counts(self):
        calls = []

        def sample_next(state, n=1):
            calls.append(n)
            return state

        model = SampleableTransitionModel(sample_next)
        invalid_counts = (
            True,
            False,
            -1,
            1.5,
            "2",
            b"2",
            np.array([2]),
            np.array(True, dtype=object),
            np.array("2"),
        )

        for n in invalid_counts:
            with self.subTest(n=n):
                with self.assertRaisesRegex(ValueError, "n must be"):
                    sample_next_state(model, array([0.0]), n=n)

        self.assertEqual(calls, [])

    def test_sampleable_transition_model_normalizes_numpy_integer_count(self):
        calls = []

        def sample_next(state, n=1):
            calls.append(n)
            return state

        model = SampleableTransitionModel(sample_next)

        sample_next_state(model, array([0.0]), n=np.array(2, dtype=np.int64))

        self.assertEqual(calls, [2])
        self.assertIs(type(calls[0]), int)

    def test_density_transition_model_rejects_invalid_sample_counts(self):
        calls = []

        def transition_density(state_next, state_previous):
            return 1.0 if state_next is state_previous else 0.0

        def sample_next(state, n=1):
            calls.append(n)
            return state

        model = DensityTransitionModel(transition_density, sample_next=sample_next)

        for n in (True, -1, "2", np.array([2])):
            with self.subTest(n=n):
                with self.assertRaisesRegex(ValueError, "n must be"):
                    model.sample_next(array([0.0]), n=n)

        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
