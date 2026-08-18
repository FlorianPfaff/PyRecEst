import unittest

import numpy as np

from pyrecest.filters.track_manager import TrackManager


class TrackManagerBooleanControlTest(unittest.TestCase):
    def test_boolean_controls_accept_numpy_boolean_scalars(self):
        manager = TrackManager(
            allow_births=np.bool_(False),
            extract_confirmed_only=np.bool_(False),
            keep_history=np.bool_(False),
        )

        self.assertFalse(manager.allow_births)
        self.assertFalse(manager.extract_confirmed_only)
        self.assertFalse(manager.keep_history)

    def test_boolean_controls_reject_truthy_non_boolean_values(self):
        for control in ("allow_births", "extract_confirmed_only", "keep_history"):
            with self.subTest(control=control):
                with self.assertRaisesRegex(ValueError, rf"{control} must be"):
                    TrackManager(**{control: "False"})

    def test_boolean_controls_reject_numeric_values(self):
        for control in ("allow_births", "extract_confirmed_only", "keep_history"):
            for value in (0, 1, np.array(0), np.array(1)):
                with self.subTest(control=control, value=value):
                    with self.assertRaisesRegex(ValueError, rf"{control} must be"):
                        TrackManager(**{control: value})
