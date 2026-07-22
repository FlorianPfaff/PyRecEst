import unittest

import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array, to_numpy
from pyrecest.filters.bingham_filter import BinghamFilter


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",
    reason="BinghamFilter is not supported on the JAX backend",
)
class BinghamFilterConjugateCopyTest(unittest.TestCase):
    def test_conjugate_does_not_mutate_input_storage(self):
        quaternion = array([1.0, 2.0, 3.0, 4.0])

        conjugate = BinghamFilter._conjugate(quaternion)

        npt.assert_array_equal(to_numpy(quaternion), [1.0, 2.0, 3.0, 4.0])
        npt.assert_array_equal(to_numpy(conjugate), [1.0, -2.0, -3.0, -4.0])


if __name__ == "__main__":
    unittest.main()
