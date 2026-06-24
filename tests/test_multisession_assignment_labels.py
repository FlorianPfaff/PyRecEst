"""Regression tests for multi-session label conversion."""

import unittest

import numpy as np
import pyrecest.utils.multisession_assignment as multisession_assignment_module
from pyrecest.backend import (  # pylint: disable=no-name-in-module
    __backend_name__,
    array,
    array_equal,
)
from pyrecest.utils import MultiSessionAssignmentResult, tracks_to_session_labels


class TestMultiSessionAssignmentLabels(unittest.TestCase):
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
    def test_duplicate_detection_rejected_when_fill_value_matches_track_label(self):
        tracks = [{0: 0}, {0: 0}]

        for name, converter in self._converters():
            with self.subTest(converter=name):
                with self.assertRaisesRegex(
                    ValueError,
                    "Each detection can only belong to a single track",
                ):
                    converter(tracks, session_sizes=[1], fill_value=-99)

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_integer_like_fill_values_are_accepted(self):
        tracks = [{0: 0}, {0: 1}]
        valid_fill_values = (-2.0, np.int64(-3), np.array(-4))

        for fill_value in valid_fill_values:
            expected_fill_value = int(np.asarray(fill_value).item())
            for name, converter in self._converters():
                with self.subTest(converter=name, fill_value=repr(fill_value)):
                    labels = converter(tracks, session_sizes=[3], fill_value=fill_value)
                    self.assertTrue(
                        array_equal(
                            labels[0], array([0, 1, expected_fill_value], dtype=int)
                        )
                    )

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_fill_value_cannot_collide_with_track_labels(self):
        tracks = [{0: 0}, {0: 1}]

        for name, converter in self._converters():
            with self.subTest(converter=name):
                with self.assertRaisesRegex(
                    ValueError,
                    "fill_value must not collide with track labels",
                ):
                    converter(tracks, session_sizes=[3], fill_value=0)


if __name__ == "__main__":
    unittest.main()
