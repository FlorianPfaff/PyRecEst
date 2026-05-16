import unittest

import numpy.testing as npt

from pyrecest.backend import array, pi, random, sum
from pyrecest.distributions import (
    CircularDiracDistribution,
    HypertoroidalDiracDistribution,
)


class TestHypertoroidalDiracMarginalizeOut(unittest.TestCase):
    @staticmethod
    def get_distribution(dim=3):
        random.seed(0)
        n = 20
        d = 2.0 * pi * random.uniform(size=(n, dim))
        w = random.uniform(size=n)
        w = w / sum(w)
        return HypertoroidalDiracDistribution(d, w)

    def test_list_dimensions_can_leave_one_dimension(self):
        hwd = self.get_distribution(3)

        wd = hwd.marginalize_out([0, 2])

        self.assertIsInstance(wd, CircularDiracDistribution)
        npt.assert_array_almost_equal(wd.d, hwd.d[:, 1])
        npt.assert_array_almost_equal(wd.w, hwd.w)

    def test_list_dimensions_can_leave_multiple_dimensions(self):
        hwd = self.get_distribution(4)

        marginalized = hwd.marginalize_out([1, 3])

        self.assertIsInstance(marginalized, HypertoroidalDiracDistribution)
        self.assertEqual(marginalized.dim, 2)
        npt.assert_array_almost_equal(marginalized.d, hwd.d[:, array([0, 2])])
        npt.assert_array_almost_equal(marginalized.w, hwd.w)

    def test_invalid_dimensions_are_rejected(self):
        hwd = self.get_distribution(3)

        with self.assertRaises(ValueError):
            hwd.marginalize_out([0, 0])

        with self.assertRaises(ValueError):
            hwd.marginalize_out(3)

        with self.assertRaises(ValueError):
            hwd.marginalize_out([0, 1, 2])


if __name__ == "__main__":
    unittest.main()
