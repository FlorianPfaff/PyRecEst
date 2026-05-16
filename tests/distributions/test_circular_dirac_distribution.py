import unittest

import numpy.testing as npt
from pyrecest.backend import array, ones, pi
from pyrecest.distributions import CircularDiracDistribution


class TestCircularDiracDistribution(unittest.TestCase):
    def test_column_vector_locations_are_stored_flat(self):
        d = array([[0.0], [pi / 2.0], [pi]])
        w = ones(3) / 3.0

        wd = CircularDiracDistribution(d, w)

        self.assertEqual(wd.d.shape, (3,))
        npt.assert_allclose(wd.d, d[:, 0])
        npt.assert_allclose(wd.w, w)

    def test_column_vector_locations_with_uniform_weights_are_stored_flat(self):
        d = array([[0.0], [pi / 2.0], [pi]])

        wd = CircularDiracDistribution(d)

        self.assertEqual(wd.d.shape, (3,))
        self.assertEqual(wd.w.shape, (3,))
        npt.assert_allclose(wd.d, d[:, 0])


if __name__ == "__main__":
    unittest.main()
