"""Regression tests for masked PyTorch split controls."""

import unittest

import numpy as np

from pyrecest.backend_support._pytorch_split_index_contract import (
    _normalize_split_cut_indices,
    _normalize_split_section_count,
)


class _TorchStub:
    @staticmethod
    def is_tensor(value):
        del value
        return False


class PytorchSplitMaskContractTest(unittest.TestCase):
    def test_rejects_masked_section_counts(self):
        masked_counts = (
            np.ma.array(3, mask=True),
            np.ma.masked,
        )

        for masked_count in masked_counts:
            with self.subTest(masked_count=masked_count):
                with self.assertRaisesRegex(
                    TypeError,
                    "slice indices must be integers",
                ):
                    _normalize_split_section_count(masked_count, _TorchStub)

    def test_rejects_masked_cut_indices(self):
        masked_indices = (
            np.ma.array([1, 3], mask=[False, True]),
            [1, np.ma.masked],
        )

        for indices in masked_indices:
            with self.subTest(indices=indices):
                with self.assertRaisesRegex(
                    TypeError,
                    "slice indices must be integers",
                ):
                    _normalize_split_cut_indices(indices, _TorchStub)

    def test_preserves_clear_mask_integer_inputs(self):
        section_count = _normalize_split_section_count(
            np.ma.array(3, mask=False),
            _TorchStub,
        )
        cut_indices = _normalize_split_cut_indices(
            np.ma.array([1, 3], mask=False),
            _TorchStub,
        )

        self.assertEqual(section_count, 3)
        self.assertEqual(cut_indices, (1, 3))


if __name__ == "__main__":
    unittest.main()
