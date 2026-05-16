import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, mod, pi
from pyrecest.distributions import HypertoroidalWNDistribution


class TestHypertoroidalWNDistribution(unittest.TestCase):
    def test_pdf(self):
        mu = array([1, 2])
        C = array([[0.5, 0.1], [0.1, 0.3]])

        hwn = HypertoroidalWNDistribution(mu, C)

        xa = array([[0, 1, 2], [1, 2, 3]]).T
        pdf_values = hwn.pdf(xa)

        expected_values = array(
            [0.0499028191873498, 0.425359477472412, 0.0499028191873498]
        )
        npt.assert_allclose(pdf_values, expected_values, rtol=2e-6)

    def test_shift_returns_copy_without_mutating_original(self):
        mu = array([1.0, 2.0])
        C = array([[0.5, 0.1], [0.1, 0.6]])
        shift_by = array([0.25, 2.0 * pi - 0.5])
        dist = HypertoroidalWNDistribution(mu, C)

        shifted = dist.shift(shift_by)

        self.assertIsNot(shifted, dist)
        npt.assert_allclose(dist.mu, mu)
        npt.assert_allclose(shifted.mu, mod(mu + shift_by, 2.0 * pi))
        npt.assert_allclose(shifted.C, dist.C)


if __name__ == "__main__":
    unittest.main()
