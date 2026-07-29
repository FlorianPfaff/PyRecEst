"""Regression tests for dense multi-session label fill-value ranges."""

import unittest

import numpy as np
import pyrecest.utils.multisession_assignment as multisession_assignment_module
from pyrecest.backend import __backend_name__
from pyrecest.utils import MultiSessionAssignmentResult, tracks_to_session_labels


class TestMultiSessionAssignmentFillValueRange(unittest.TestCase):
    @staticmethod
    def _converters():
        return (
            ("public", tracks_to_session_labels),
            ("module", multisession_assignment_module.tracks_to_session_labels),
            (
                "result_method",
                lambda track_list, **kwargs: MultiSessionAssignmentResult(
                    tracks=track_list,
                    matched_edges=[],
                    total_cost=0.0,
                ).to_session_labels(**kwargs),
            ),
        )

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_fill_values_outside_int64_range_are_rejected(self):
        int64_info = np.iinfo(np.int64)
        invalid_fill_values = (
            int(int64_info.max) + 1,
            int(int64_info.min) - 1,
            np.uint64(int64_info.max) + np.uint64(1),
        )

        for fill_value in invalid_fill_values:
            for name, converter in self._converters():
                with self.subTest(converter=name, fill_value=repr(fill_value)):
                    with self.assertRaisesRegex(ValueError, "fit in int64"):
                        converter([], session_sizes=[1], fill_value=fill_value)

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_int64_boundary_fill_values_are_preserved(self):
        int64_info = np.iinfo(np.int64)
        valid_fill_values = (
            int(int64_info.min),
            int(int64_info.max),
            np.int64(int64_info.min),
            np.uint64(int64_info.max),
        )

        for fill_value in valid_fill_values:
            for name, converter in self._converters():
                with self.subTest(converter=name, fill_value=repr(fill_value)):
                    labels = converter([], session_sizes=[1], fill_value=fill_value)
                    self.assertEqual(int(labels[0][0]), int(fill_value))


if __name__ == "__main__":
    unittest.main()
