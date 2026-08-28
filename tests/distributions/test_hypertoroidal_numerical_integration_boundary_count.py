import unittest

import pyrecest.backend
from pyrecest.backend import array
from pyrecest.distributions.circle.wrapped_normal_distribution import (
    WrappedNormalDistribution,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Numerical hypertoroidal integration is supported only on NumPy.",
)
class TestHypertoroidalNumericalIntegrationBoundaryCount(unittest.TestCase):
    def test_one_dimensional_distribution_rejects_extra_boundary_rows(self):
        dist = WrappedNormalDistribution(array(0.0), array(1.0))

        with self.assertRaisesRegex(ValueError, "one row per dimension"):
            dist.integrate_numerically(array([[0.0, 1.0], [0.0, 1.0]]))


if __name__ == "__main__":
    unittest.main()
