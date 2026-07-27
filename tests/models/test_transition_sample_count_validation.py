"""Regression tests for transition-sampler sample-count validation."""

import unittest

import numpy as np
from pyrecest.backend import array
from pyrecest.models import (
    DensityTransitionModel,
    SampleableTransitionModel,
    sample_next_state,
)

_TEMPORAL_COUNTS = (
    np.timedelta64(2, "ns"),
    np.datetime64("1970-01-01T00:00:00.000000002", "ns"),
    np.array(np.timedelta64(2, "ns"), dtype=object),
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
            np.ma.array(2, mask=True),
        ) + _TEMPORAL_COUNTS

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
        sample_next_state(model, array([0.0]), n=np.ma.array(3, mask=False))

        self.assertEqual(calls, [2, 3])
        self.assertTrue(all(type(count) is int for count in calls))

    def test_density_transition_model_rejects_invalid_sample_counts(self):
        calls = []

        def transition_density(state_next, state_previous):
            return 1.0 if state_next is state_previous else 0.0

        def sample_next(state, n=1):
            calls.append(n)
            return state

        model = DensityTransitionModel(transition_density, sample_next=sample_next)

        invalid_counts = (
            True,
            -1,
            "2",
            np.array([2]),
            np.ma.array(2, mask=True),
        ) + _TEMPORAL_COUNTS
        for n in invalid_counts:
            with self.subTest(n=n):
                with self.assertRaisesRegex(ValueError, "n must be"):
                    model.sample_next(array([0.0]), n=n)

        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
