import unittest
from fractions import Fraction

import numpy as np
from pyrecest.backend_support._pytorch_split_index_contract import (
    _normalize_split_section_count,
)


class _TorchStub:
    @staticmethod
    def is_tensor(value):
        del value
        return False


class PytorchSplitIndexContractTest(unittest.TestCase):
    def test_rejects_fractional_count_hidden_by_float_rounding(self):
        fractional_count = Fraction(2**54 + 1, 2)

        with self.assertRaisesRegex(ValueError, "number sections must be an integer"):
            _normalize_split_section_count(fractional_count, _TorchStub)

    def test_preserves_large_exact_integral_count(self):
        exact_count = Fraction(2**54 + 2, 2)

        normalized = _normalize_split_section_count(exact_count, _TorchStub)

        self.assertEqual(normalized, 2**53 + 1)

    def test_rejects_temporal_section_counts(self):
        temporal_counts = (
            np.timedelta64(2, "ns"),
            np.timedelta64(2, "us"),
            np.datetime64("1970-01-01T00:00:00.000000002"),
            np.array(np.timedelta64(2, "ns"), dtype=object),
        )

        for temporal_count in temporal_counts:
            with self.subTest(temporal_count=temporal_count):
                with self.assertRaisesRegex(
                    TypeError,
                    "slice indices must be integers",
                ):
                    _normalize_split_section_count(temporal_count, _TorchStub)

    def test_preserves_numpy_integer_count(self):
        normalized = _normalize_split_section_count(np.int64(3), _TorchStub)

        self.assertEqual(normalized, 3)


if __name__ == "__main__":
    unittest.main()
