import unittest

import numpy as np
import pyrecest.backend
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import (
    HypertoroidalFourierDistribution,
)
from pyrecest.distributions.hypertorus.low_rank_hypertoroidal_fourier_distribution import (
    LowRankHypertoroidalFourierDistribution,
)


class _LowRankSubclass(LowRankHypertoroidalFourierDistribution):
    pass


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",  # pylint: disable=no-member
    reason="Low-rank Fourier prototype is NumPy-only",
)
class TestLowRankFourierSubclassPreservation(unittest.TestCase):
    def setUp(self):
        self.dist = _LowRankSubclass.uniform((3, 3), transformation="identity")
        self.dense = HypertoroidalFourierDistribution(
            np.asarray(self.dist.to_dense()), "identity"
        )

    def test_shift_and_hermitian_repair_preserve_subclass(self):
        shifted = self.dist.shift(np.array([0.2, -0.3]))
        repaired = self.dist.centered_hermitianized()

        self.assertIs(type(shifted), _LowRankSubclass)
        self.assertIs(type(repaired), _LowRankSubclass)

    def test_binary_operations_preserve_left_operand_subclass(self):
        results = (
            self.dist.multiply(self.dist),
            self.dist.multiply(self.dense),
            self.dist.convolve(self.dist),
            self.dist.convolve(self.dense),
        )

        for result in results:
            with self.subTest(operation=type(result).__name__):
                self.assertIs(type(result), _LowRankSubclass)


if __name__ == "__main__":
    unittest.main()
