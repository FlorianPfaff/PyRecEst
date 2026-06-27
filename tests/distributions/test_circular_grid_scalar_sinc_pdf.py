import unittest

from pyrecest.backend import array, pi
from pyrecest.distributions.circle.circular_grid_distribution import (
    CircularGridDistribution,
)


class CircularGridScalarSincPdfTest(unittest.TestCase):
    def test_pdf_via_sinc_accepts_scalar_angle(self):
        dist = CircularGridDistribution.from_function(
            lambda xs: xs * 0.0 + 1.0 / (2.0 * pi),
            9,
        )

        scalar_value = dist.pdf(0.0, use_sinc=True, sinc_repetitions=3)
        vector_value = dist.pdf(array([0.0]), use_sinc=True, sinc_repetitions=3)[0]

        self.assertAlmostEqual(float(scalar_value), float(vector_value))

    def test_sinc_interpolation_matches_knots(self):
        values = array([0.05, 0.15, 0.35, 0.25, 0.2])
        dist = CircularGridDistribution(values, enforce_pdf_nonnegative=False)
        evaluated = dist.pdf(dist.get_grid(), use_sinc=True, sinc_repetitions=3)

        for actual, expected in zip(evaluated, values):
            self.assertAlmostEqual(float(actual), float(expected), delta=1e-7)


if __name__ == "__main__":
    unittest.main()
