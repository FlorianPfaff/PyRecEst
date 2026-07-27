import math
import unittest

from pyrecest.backend import array
from pyrecest.distributions import SO3BinghamDistribution
from tests.distributions.so3_test_helpers import scalar


class SO3BinghamLogPdfStabilityTest(unittest.TestCase):
    def test_log_pdf_remains_finite_when_pdf_underflows(self):
        concentration = 1000.0
        distribution = SO3BinghamDistribution.from_mode_and_concentration(
            array([0.0, 0.0, 0.0, 1.0]), concentration
        )
        orthogonal_quaternion = array([1.0, 0.0, 0.0, 0.0])

        self.assertEqual(scalar(distribution.pdf(orthogonal_quaternion)), 0.0)

        log_density = scalar(distribution.ln_pdf(orthogonal_quaternion))
        expected = (
            math.log(2.0)
            - math.log(scalar(distribution.distFullSphere.F))
            - concentration
        )

        self.assertTrue(math.isfinite(log_density))
        self.assertAlmostEqual(log_density, expected, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
