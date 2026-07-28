"""Regression tests for nonlinear motion-model turn thresholds."""

import unittest

import numpy as np
from pyrecest.backend import array
from pyrecest.models import coordinated_turn_transition, se2_unicycle_transition


class TestTurnThresholdValidation(unittest.TestCase):
    def test_nonlinear_turn_transitions_reject_invalid_thresholds(self):
        transitions = (
            (
                coordinated_turn_transition,
                array([0.0, 0.0, 1.0, 0.0, 0.0]),
            ),
            (
                se2_unicycle_transition,
                array([0.0, 0.0, 0.0, 1.0, 0.0]),
            ),
        )
        invalid_thresholds = (0.0, -1e-8, np.nan, np.inf, True, "1e-8", b"1e-8")

        for transition, state in transitions:
            for turn_threshold in invalid_thresholds:
                with self.subTest(
                    transition=transition.__name__, turn_threshold=turn_threshold
                ):
                    with self.assertRaisesRegex(ValueError, "turn_threshold"):
                        transition(state, turn_threshold=turn_threshold)


if __name__ == "__main__":
    unittest.main()
