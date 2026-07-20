import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions.nonperiodic.hyperrectangular_uniform_distribution import (
    HyperrectangularUniformDistribution,
)


class TestHyperrectangularUniformDistribution(unittest.TestCase):
    def test_pdf_is_zero_outside_bounds(self):
        dist = HyperrectangularUniformDistribution(array([[0.0, 1.0], [10.0, 12.0]]))

        pdf_values = dist.pdf(
            array(
                [
                    [0.5, 11.0],
                    [-0.5, 11.0],
                    [0.5, 20.0],
                    [1.0, 10.0],
                ]
            )
        )

        npt.assert_allclose(pdf_values, array([0.5, 0.0, 0.0, 0.5]))

    def test_pdf_accepts_single_multidimensional_point(self):
        dist = HyperrectangularUniformDistribution(array([[0.0, 1.0], [10.0, 12.0]]))

        npt.assert_allclose(float(dist.pdf(array([0.5, 11.0]))), 0.5)

    def test_pdf_rejects_complex_points(self):
        dist = HyperrectangularUniformDistribution(array([[0.0, 1.0], [10.0, 12.0]]))
        invalid_points = (
            array([0.5 + 0.25j, 11.0]),
            array([[0.5 + 0.25j, 11.0]]),
        )

        for points in invalid_points:
            with self.subTest(points=points):
                with self.assertRaisesRegex(ValueError, "real-valued"):
                    dist.pdf(points)

    def test_pdf_rejects_nonfinite_points(self):
        dist = HyperrectangularUniformDistribution(array([[0.0, 1.0], [10.0, 12.0]]))
        invalid_points = (
            array([0.5, float("nan")]),
            array([float("inf"), 11.0]),
            array([[0.5, 11.0], [-float("inf"), 11.0]]),
        )

        for points in invalid_points:
            with self.subTest(points=points):
                with self.assertRaisesRegex(ValueError, "finite"):
                    dist.pdf(points)

    def test_pdf_rejects_wrong_point_dimension(self):
        dist = HyperrectangularUniformDistribution(array([[0.0, 1.0], [10.0, 12.0]]))

        with self.assertRaises(ValueError):
            dist.pdf(array([0.5, 11.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
