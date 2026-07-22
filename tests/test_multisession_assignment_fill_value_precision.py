"""Regression test for exact multi-session label fill values."""

import unittest
from decimal import Decimal

from pyrecest.backend import __backend_name__
from pyrecest.utils import tracks_to_session_labels


class TestMultiSessionAssignmentFillValuePrecision(unittest.TestCase):
    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_exact_decimal_fill_value_is_not_rounded_through_float(self):
        fill_value = Decimal(-(2**53 + 1))

        labels = tracks_to_session_labels(
            [{0: 0}],
            session_sizes=[2],
            fill_value=fill_value,
        )

        self.assertEqual(int(labels[0][1]), int(fill_value))


if __name__ == "__main__":
    unittest.main()
