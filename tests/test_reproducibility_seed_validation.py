"""Regression tests for reproducibility seed validation."""

from __future__ import annotations

import unittest

import numpy as np
from pyrecest.reproducibility import _normalize_seed


class TestReproducibilitySeedValidation(unittest.TestCase):
    def test_temporal_seed_scalars_are_rejected_before_item_unwrap(self) -> None:
        invalid_values = (
            np.timedelta64(0, "ns"),
            np.timedelta64(1, "ns"),
            np.datetime64("1970-01-01T00:00:00.000000000"),
            np.datetime64("1970-01-01T00:00:00.000000001"),
            np.array(np.timedelta64(1, "ns")),
            np.array(np.datetime64("1970-01-01T00:00:00.000000001")),
            np.array(np.timedelta64(1, "ns"), dtype=object),
            np.array(
                np.datetime64("1970-01-01T00:00:00.000000001"),
                dtype=object,
            ),
        )

        for invalid_value in invalid_values:
            with self.subTest(invalid_value=repr(invalid_value)):
                with self.assertRaisesRegex(
                    ValueError,
                    "seed must be a non-negative integer or None",
                ):
                    _normalize_seed(invalid_value)

    def test_masked_seed_scalars_are_rejected_before_item_unwrap(self) -> None:
        invalid_values = (
            np.ma.masked,
            np.ma.array(7, mask=True),
        )

        for invalid_value in invalid_values:
            with self.subTest(invalid_value=repr(invalid_value)):
                with self.assertRaisesRegex(
                    ValueError,
                    "seed must be a non-negative integer or None",
                ):
                    _normalize_seed(invalid_value)

    def test_numeric_scalar_arrays_are_still_valid_seeds(self) -> None:
        self.assertEqual(_normalize_seed(np.array(0, dtype=np.int64)), 0)
        self.assertEqual(_normalize_seed(np.array(1.0, dtype=np.float64)), 1)
        self.assertEqual(_normalize_seed(np.ma.array(7, mask=False)), 7)
        self.assertIsNone(_normalize_seed(None))


if __name__ == "__main__":
    unittest.main()
