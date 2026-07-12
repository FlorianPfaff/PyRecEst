import unittest

import numpy.testing as npt

import pyrecest.backend
from pyrecest.backend import array, to_numpy
from pyrecest.filters.bingham_filter import BinghamFilter


class TestBinghamFilterConjugateIndependence(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="BinghamFilter is not supported on the JAX backend",
    )
    def test_conjugate_does_not_mutate_input(self):
        quaternion = array([1.0, 2.0, 3.0, 4.0])

        conjugated = BinghamFilter._conjugate(quaternion)

        npt.assert_allclose(to_numpy(quaternion), [1.0, 2.0, 3.0, 4.0])
        npt.assert_allclose(to_numpy(conjugated), [1.0, -2.0, -3.0, -4.0])


if __name__ == "__main__":
    unittest.main()
