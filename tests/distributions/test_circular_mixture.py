import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions import CircularMixture, VonMisesDistribution


class TestCircularMixture(unittest.TestCase):
    def setUp(self):
        self.dist1 = VonMisesDistribution(0.2, 1.5)
        self.dist2 = VonMisesDistribution(1.1, 0.7)
        self.weights = array([0.25, 0.75])
        self.mixture = CircularMixture([self.dist1, self.dist2], self.weights)

    def expected_pdf(self, xs):
        return self.weights[0] * self.dist1.pdf(xs) + self.weights[1] * self.dist2.pdf(xs)

    def test_pdf_accepts_vector_of_angles(self):
        xs = array([0.0, 0.5, 1.0])

        actual = self.mixture.pdf(xs)
        expected = self.expected_pdf(xs)

        npt.assert_allclose(actual, expected)

    def test_pdf_accepts_column_vector_of_angles(self):
        xs = array([0.0, 0.5, 1.0])
        xs_col = array([[0.0], [0.5], [1.0]])

        actual = self.mixture.pdf(xs_col)
        expected = self.expected_pdf(xs)

        npt.assert_allclose(actual, expected)

    def test_pdf_accepts_scalar_angle(self):
        x = array(0.5)

        actual = self.mixture.pdf(x)
        expected = self.expected_pdf(x)

        npt.assert_allclose(actual, expected)

    def test_pdf_rejects_row_vector(self):
        with self.assertRaises(AssertionError):
            self.mixture.pdf(array([[0.0, 0.5, 1.0]]))


if __name__ == "__main__":
    unittest.main()
