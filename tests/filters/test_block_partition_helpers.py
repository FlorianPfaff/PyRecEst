import unittest

from pyrecest.filters import contiguous_partition, resolve_partition, validate_partition


class BlockPartitionHelpersTest(unittest.TestCase):
    def test_contiguous_partition_covers_components(self):
        self.assertEqual(
            contiguous_partition(5, block_size=2),
            ((0, 1), (2, 3), (4,)),
        )

    def test_resolve_named_partitions(self):
        self.assertEqual(resolve_partition(3, None), ((0, 1, 2),))
        self.assertEqual(resolve_partition(3, "singleton"), ((0,), (1,), (2,)))
        self.assertEqual(resolve_partition(5, "contiguous"), ((0, 1, 2, 3), (4,)))

    def test_validate_partition_rejects_overlap_and_gaps(self):
        self.assertEqual(validate_partition([(0, 2), (1,)], 3), ((0, 2), (1,)))
        with self.assertRaises(ValueError):
            validate_partition([(0, 1), (1, 2)], 3)
        with self.assertRaises(ValueError):
            validate_partition([(0,)], 3)


if __name__ == "__main__":
    unittest.main()
