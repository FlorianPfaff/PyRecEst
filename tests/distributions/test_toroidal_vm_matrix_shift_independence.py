import unittest

import pyrecest.backend
from pyrecest.backend import array
from pyrecest.distributions.hypertorus.toroidal_vm_matrix_distribution import (
    ToroidalVMMatrixDistribution,
)


class TestToroidalVMMatrixShiftIndependence(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="JAX arrays are immutable and cannot expose mutable aliasing.",
    )
    def test_shifted_parameters_do_not_alias_original_distribution(self):
        original = ToroidalVMMatrixDistribution(
            array([1.0, 2.0]),
            array([0.5, 0.7]),
            array([[0.3, 0.1], [-0.2, 0.4]]),
        )

        shifted = original.shift(array([0.5, -0.3]))
        shifted.kappa[0] = 9.0
        shifted.A[0, 0] = 9.0

        self.assertAlmostEqual(float(original.kappa[0]), 0.5)
        self.assertAlmostEqual(float(original.A[0, 0]), 0.3)
        self.assertAlmostEqual(float(shifted.kappa[0]), 9.0)
        self.assertAlmostEqual(float(shifted.A[0, 0]), 9.0)


if __name__ == "__main__":
    unittest.main()
