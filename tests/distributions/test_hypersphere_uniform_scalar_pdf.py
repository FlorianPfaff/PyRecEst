"""Regression tests for hyperspherical uniform input validation."""

import unittest

from pyrecest.distributions import HypersphericalUniformDistribution


class HypersphericalUniformScalarPdfTest(unittest.TestCase):
    def test_pdf_rejects_scalar_input_with_value_error(self):
        dist = HypersphericalUniformDistribution(2)

        with self.assertRaisesRegex(ValueError, "Invalid shape"):
            dist.pdf(1.0)


if __name__ == "__main__":
    unittest.main()
